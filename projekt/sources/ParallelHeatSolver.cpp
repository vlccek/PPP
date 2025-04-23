/**
 * @file    ParallelHeatSolver.cpp
 *
 * @author  Jakub Vlk <xvlkja07@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 *          Disclaimer -- contains "hovnokod"
 *
 * @date    2024-02-23
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <ios>
#include <string_view>

#include "ParallelHeatSolver.hpp"

ParallelHeatSolver::ParallelHeatSolver(const SimulationProperties& simulationProps,
                                       const MaterialProperties& materialProps)
    : HeatSolverBase(simulationProps, materialProps) {
    MPI_Comm_size(MPI_COMM_WORLD, &mWorldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mWorldRank);

    /**********************************************************************************************************************/
    /*                                  Call init* and alloc* methods in correct order                                    */
    /**********************************************************************************************************************/

    initGridTopology();
    initDataDistribution();
    allocLocalTiles();
    initHaloExchange();

    if (!mSimulationProps.getOutputFileName().empty()) {
        /**********************************************************************************************************************/
        /*                               Open output file if output file name was specified.                                  */
        /*  If mSimulationProps.useParallelIO() flag is set to true, open output file for parallel access, otherwise open it  */
        /*                         only on MASTER rank using sequetial IO. Use openOutputFile* methods.                       */
        /**********************************************************************************************************************/

        if (mSimulationProps.useParallelIO()) {
            openOutputFileParallel();
        }
        else {
            if (mWorldRank == 0) {
                mFileHandle = H5Fcreate(simulationProps.getOutputFileName().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                        H5P_DEFAULT);
            }
        }
    }
}

ParallelHeatSolver::~ParallelHeatSolver() {
    /**********************************************************************************************************************/
    /*                                  Call deinit* and dealloc* methods in correct order                                */
    /*                                             (should be in reverse order)                                           */
    /**********************************************************************************************************************/

    deinitHaloExchange();
    deallocLocalTiles();
    deinitDataDistribution();
    deinitGridTopology();
}

std::string_view ParallelHeatSolver::getCodeType() const {
    return codeType;
}

void ParallelHeatSolver::initGridTopology() {
    /**********************************************************************************************************************/
    /*                          Initialize 2D grid topology using non-periodic MPI Cartesian topology.                    */
    /*                       Also create a communicator for middle column average temperature computation.                */
    /**********************************************************************************************************************/
    const int ndims = 2;
    int nX, nY;
    mSimulationProps.getDecompGrid(nX, nY);
    int periods[ndims] = {0, 0};

    int dims[ndims] = {nX, nY};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &gridComm); // reorder for better performance
    MPI_Comm_set_name(gridComm, "GridComm");

    int my_grid_rank;
    MPI_Comm_rank(gridComm, &my_grid_rank);

    MPI_Cart_coords(gridComm, my_grid_rank, ndims, myCoorsGrid);

    MPI_Comm_size(gridComm, &m_size);

    int middle_col_index = dims[0] / 2;
    int color = (myCoorsGrid[0] == middle_col_index) ? 1 : MPI_UNDEFINED;

    int key = myCoorsGrid[1];

    // Rozdělení komunikátoru gridComm
    MPI_Comm_split(gridComm, color, key, &avgTempComm);


#ifdef USE_KAMPING_LIB
     if (avgTempComm != MPI_COMM_NULL) {
        m_kamping_avg_comm = kamping::Communicator(avgTempComm);
     }
#endif
}

void ParallelHeatSolver::deinitGridTopology() {
    /**********************************************************************************************************************/
    /*      Deinitialize 2D grid topology and the middle column average temperature computation communicator              */
    /**********************************************************************************************************************/
#ifndef USE_KAMPING_LIB
    MPI_Comm_free(&gridComm);
#endif
    if (avgTempComm != MPI_COMM_NULL) {
        MPI_Comm_free(&avgTempComm);
    }
}

void ParallelHeatSolver::initDataDistribution() {
    /**********************************************************************************************************************/
    /*                 Initialize variables and MPI datatypes for data distribution (float and int).                      */
    /**********************************************************************************************************************/
    mGridSizeEdge = mMaterialProps.getEdgeSize();

    int nX, nY;
    mSimulationProps.getDecompGrid(nX, nY);

    mLocalTileSize[0] = mGridSizeEdge / nX;
    mLocalTileSize[1] = mGridSizeEdge / nY;

    // mby add hallo but probably no ????
    int local_tile_width = mLocalTileSize[0];
    int local_tile_height = mLocalTileSize[1];
    int global_grid_width = mMaterialProps.getEdgeSize();


    MPI_Datatype tile_layout_type_float;
    MPI_Type_vector(local_tile_height, local_tile_width, global_grid_width, MPI_FLOAT, &tile_layout_type_float);

    MPI_Aint float_lb, float_extent;
    MPI_Type_get_extent(MPI_FLOAT, &float_lb, &float_extent);
    MPI_Type_create_resized(tile_layout_type_float, 0, float_extent, &tileTypeFloat);

    MPI_Type_free(&tile_layout_type_float); // Uvolnit dočasný layout
    MPI_Type_commit(&tileTypeFloat); // Commit finálního resized typu

    // --- INT ---
    MPI_Datatype tile_layout_type_int; // Přejmenováno pro jasnost
    MPI_Type_vector(local_tile_height, local_tile_width, global_grid_width, MPI_INT, &tile_layout_type_int);

    MPI_Aint int_lb, int_extent;
    MPI_Type_get_extent(MPI_INT, &int_lb, &int_extent);
    MPI_Type_create_resized(tile_layout_type_int, 0, int_extent, &tileTypeInt);

    MPI_Type_free(&tile_layout_type_int); // Uvolnit dočasný layout
    MPI_Type_commit(&tileTypeInt); // Commit finálního resized typu


    int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;

    // Vytvoření a commit pro FLOAT
    MPI_Type_vector(mLocalTileSize[1], mLocalTileSize[0], localBufferWidth, MPI_FLOAT,
                    &m_localActiveAreaTypeFloat); // Použijte správný handle
    MPI_Type_commit(&m_localActiveAreaTypeFloat); // Commit správného handle

    // Vytvoření a commit pro INT
    MPI_Type_vector(mLocalTileSize[1], mLocalTileSize[0], localBufferWidth, MPI_INT,
                    &m_localActiveAreaTypeInt); // Použijte druhý, správný handle
    MPI_Type_commit(&m_localActiveAreaTypeInt);
}

void ParallelHeatSolver::deinitDataDistribution() {
    /**********************************************************************************************************************/
    /*                       Deinitialize variables and MPI datatypes for data distribution.                              */
    /**********************************************************************************************************************/

    MPI_Type_free(&tileTypeFloat);
    MPI_Type_free(&tileTypeInt);
}

void ParallelHeatSolver::allocLocalTiles() {
    /**********************************************************************************************************************/
    /*            Allocate local tiles for domain map (1x), domain parameters (1x) and domain temperature (2x).           */
    /*                                               Use AlignedAllocator.                                                */
    /**********************************************************************************************************************/

    int size = (mLocalTileSize[0] + 2 * haloZoneSize) * (mLocalTileSize[1] + 2 * haloZoneSize);

    mLocalTileMaterial.resize(size);
    mLocalTileMaterialProp.resize(size);


    mLocalTileTemperature[0].resize(size);
    mLocalTileTemperature[1].resize(size);
}

void ParallelHeatSolver::deallocLocalTiles() {
    /**********************************************************************************************************************/
    /*                                   Deallocate local tiles (may be empty).                                           */
    /**********************************************************************************************************************/

    // not necessary to deallocate, as std::vector will do it automatically :)
}

void ParallelHeatSolver::initHaloExchange() {
    /**********************************************************************************************************************/
    /*                            Initialize variables and MPI datatypes for halo exchange.                               */
    /*                    If mSimulationProps.isRunParallelRMA() flag is set to true, create RMA windows.                 */
    /**********************************************************************************************************************/


    int nX, nY;
    mSimulationProps.getDecompGrid(nX, nY);

    int globalEdgeSize = mMaterialProps.getEdgeSize();

    int haloSize = haloZoneSize;

    int local_active_width = globalEdgeSize / nX;
    int local_active_height = globalEdgeSize / nY;

    int local_buffer_stride = local_active_width + 2 * haloSize;

    MPI_Type_vector(
        haloSize,
        local_active_width,
        local_buffer_stride,
        MPI_FLOAT,
        &horizontal_strip_type
    );
    MPI_Type_commit(&horizontal_strip_type);

    MPI_Type_vector(
        local_active_height,
        haloSize,
        local_buffer_stride,
        MPI_FLOAT,
        &vertical_strip_type
    );
    MPI_Type_commit(&vertical_strip_type);


    MPI_Cart_shift(
        gridComm,
        0, // X axis
        1, // shift by 1
        &mLeftRank,
        &mRightRank
    );
    MPI_Cart_shift(
        gridComm,
        1, // Y axis
        1, // shift by 1
        &mTopRank,
        &mBottomRank
    );


    MPI_Aint window_size_bytes = (MPI_Aint)(local_active_width + 2 * haloZoneSize)
        * (MPI_Aint)(local_active_height + 2 * haloZoneSize)
        * sizeof(float);

    float* window_base_ptr = mLocalTileTemperature[0].data(); // Ukazatel na data vektoru

    // Vytvoření okna
    // Použijeme MPI_Win_create, protože paměť už byla alokována pomocí std::vector
    MPI_Win_create(
        window_base_ptr, // Adresa začátku paměti pro okno
        window_size_bytes, // Velikost zpřístupněné paměti v bytech
        sizeof(float), // Displacement unit - velikost základního prvku pro posunutí
        MPI_INFO_NULL, // Info objekt (nepoužíváme speciální hinty)
        gridComm, // Komunikátor asociovaný s oknem (musí být ten s topologií!)
        &m_rma_win[0] // Ukazatel na handle okna (členská proměnná)
    );

    window_base_ptr = mLocalTileTemperature[1].data(); // Ukazatel na data vektoru

    // Vytvoření okna
    // Použijeme MPI_Win_create, protože paměť už byla alokována pomocí std::vector
    MPI_Win_create(
        window_base_ptr, // Adresa začátku paměti pro okno
        window_size_bytes, // Velikost zpřístupněné paměti v bytech
        sizeof(float), // Displacement unit - velikost základního prvku pro posunutí
        MPI_INFO_NULL, // Info objekt (nepoužíváme speciální hinty)
        gridComm, // Komunikátor asociovaný s oknem (musí být ten s topologií!)
        &m_rma_win[1] // Ukazatel na handle okna (členská proměnná)
    );
}


void ParallelHeatSolver::deinitHaloExchange() {
    /**********************************************************************************************************************/
    /*                            Deinitialize variables and MPI datatypes for halo exchange.                             */
    /**********************************************************************************************************************/
    MPI_Type_free(&horizontal_strip_type);
    MPI_Type_free(&vertical_strip_type);

    if (m_rma_win[0] != MPI_WIN_NULL) {
        // Kontrola pro jistotu
        MPI_Win_free(&m_rma_win[0]);
    }

    if (m_rma_win[1] != MPI_WIN_NULL) {
        // Kontrola pro jistotu
        MPI_Win_free(&m_rma_win[1]);
    }
}

template <typename T>
void ParallelHeatSolver::localDomainMap(const T* globalData, T* localData) {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported scatter datatype!");

    /**********************************************************************************************************************/
    /*                      Implement master's global tile scatter to each rank's local tile.                             */
    /*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
    /*                                                                                                                    */
    /*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
    /*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
    /**********************************************************************************************************************/
    const MPI_Datatype resizedTileType = std::is_same_v<T, int> ? tileTypeInt : tileTypeFloat;
    const MPI_Datatype localRecvType = std::is_same_v<T, int>
                                           ? m_localActiveAreaTypeInt
                                           : m_localActiveAreaTypeFloat;


    std::vector<int> sendcounts;
    std::vector<int> displs;

    int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;
    int offset = haloZoneSize * localBufferWidth + haloZoneSize;
    // localBufferRawPtr ukazuje na začátek celého lokálního bufferu
    T* activeAreaStartPtr = localData + offset;

    // 3. Root proces připraví sendcounts a displs
    if (rank(gridComm) == ROOT) {
        sendcounts.resize(m_size);
        displs.resize(m_size);

        for (int i = 0; i < m_size; ++i) {
            // Posíláme JEDEN blok popsaný resizedTileType každému procesu
            sendcounts[i] = 1;

            // Vypočítáme posunutí pro proces 'i'
            int coords[2]; // Souřadnice procesu 'i' v procesorové mřížce
            // Je nutné použít komunikátor mřížky!
            MPI_Cart_coords(gridComm, i, 2, coords);

            int startRow = coords[1] * mLocalTileSize[1]; // Y-coord * tile_height
            int startCol = coords[0] * mLocalTileSize[0]; // X-coord * tile_width

            int globalWidth = mMaterialProps.getEdgeSize();

            // Výpočet lineárního posunutí v počtu základních prvků (protože extent resizedTileType = extent baseMpiType)
            displs[i] = startRow * globalWidth + startCol;
        }
    }


    // 4. Zavolej MPI_Scatterv
    MPI_Scatterv(
        globalData,
        sendcounts.data(),
        displs.data(),
        resizedTileType, // Typ ODESÍLANÉ jednotky (z globálu)
        activeAreaStartPtr, // Ukazatel na začátek aktivní oblasti v lokálním bufferu
        1, // !!!!! Přijímáme 1 blok tohoto nového typu !!!!!
        localRecvType, // !!!!! Typ popisující layout PŘIJÍMANÝCH dat v lokálním bufferu !!!!!
        ROOT,
        gridComm
    );
}

template <typename T>
void ParallelHeatSolver::gatherTiles(const T* localData, T* globalData) {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported gather datatype!");

    /**********************************************************************************************************************/
    /*                      Implement each rank's local tile gather to master's rank global tile.                         */
    /*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
    /*                                                                                                                    */
    /*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
    /*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
    /**********************************************************************************************************************/
    const MPI_Datatype recvTypeOnRoot = std::is_same_v<T, int> ? tileTypeInt : tileTypeFloat;
    const MPI_Datatype sendTypeFromLocal = std::is_same_v<T, int>
                                               ? m_localActiveAreaTypeInt
                                               : m_localActiveAreaTypeFloat;
    int globalWidth = mMaterialProps.getEdgeSize();

    std::vector<int> recvcounts; // Pouze na rootu
    std::vector<int> displs; // Pouze na rootu

    // Výpočet ukazatele na začátek aktivní oblasti pro odeslání
    int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;
    int offset = haloZoneSize * localBufferWidth + haloZoneSize;
    const T* activeAreaStartPtr = localData + offset; // Ukazatel pro čtení z aktivní oblasti

    if (mWorldRank == ROOT) {
        // Použijte mWorldRank nebo rank(gridComm)
        recvcounts.resize(m_size);
        displs.resize(m_size);

        for (int i = 0; i < m_size; ++i) {
            // Root přijímá JEDEN blok popsaný recvTypeOnRoot od každého procesu
            recvcounts[i] = 1;

            // Vypočítáme posunutí pro proces 'i' v globálním poli
            int coords[2];
            MPI_Cart_coords(gridComm, i, 2, coords);

            // ========= OPRAVA ZDE ==========
            // Správný výpočet počátečního řádku a sloupce
            int startRow = coords[1] * mLocalTileSize[1]; // Y * VÝŠKA lokální dlaždice
            int startCol = coords[0] * mLocalTileSize[0]; // X * ŠÍŘKA lokální dlaždice
            // ===============================

            displs[i] = startRow * globalWidth + startCol; // Výpočet lineárního posunutí
        }
    }

    // Zavolej MPI_Gatherv
    MPI_Gatherv(
        activeAreaStartPtr, // sendbuf: Ukazatel na začátek AKTIVNÍ oblasti pro odeslání
        1, // sendcount: Posíláme 1 blok typu sendTypeFromLocal
        sendTypeFromLocal, // sendtype: Typ popisující AKTIVNÍ oblast v lokálním bufferu
        globalData, // recvbuf: Buffer na rootu
        recvcounts.data(), // recvcounts: Pole počtů (1) na rootu
        displs.data(), // displs: Pole posunutí v globálním bufferu na rootu
        recvTypeOnRoot, // recvtype: Typ popisující DLAŽDICI v globálním bufferu
        ROOT, // Rank root procesu
        gridComm // Komunikátor mřížky
    );
}

void ParallelHeatSolver::computeHaloZones(const float* oldTemp, float* newTemp) {
    /**********************************************************************************************************************/
    /*  Compute new temperatures in halo zones, so that copy operations can be overlapped with inner region computation.  */
    /*                        Use updateTile method to compute new temperatures in halo zones.                            */
    /*                             TAKE CARE NOT TO COMPUTE THE SAME AREAS TWICE                                          */
    /**********************************************************************************************************************/

    const int h = haloZoneSize; // = 2
    const int W = mLocalTileSize[0]; // Šířka aktivní oblasti
    const int H = mLocalTileSize[1]; // Výška aktivní oblasti
    // Celková šířka bufferu (stride)
    const std::size_t stride = static_cast<std::size_t>(W + 2 * h);
    const float* params = mLocalTileMaterialProp.data();
    const int* map = mLocalTileMaterial.data();

    std::array<int, 4> neighbours = {mTopRank, mBottomRank, mRightRank, mLeftRank};

    int TOP = 0;
    int BOTTOM = 1;
    int RIGHT = 2;
    int LEFT = 3;


    // Horní hrana (pro odeslání nahoru)
    if (neighbours[TOP] != MPI_PROC_NULL) {
        // Počítáme řádky h až 2h-1, sloupce 2h až h+W-h-1
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + h,
                   /*offsetY*/ h,
                   /*sizeX*/ W - 2 * h, // Šířka bez levého a pravého rohu
                   /*sizeY*/ h, // Výška okraje
                   stride);
    }

    // Spodní hrana (pro odeslání dolů)
    if (neighbours[BOTTOM] != MPI_PROC_NULL) {
        // Počítáme řádky h+H-h až h+H-1, sloupce 2h až h+W-h-1
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + h,
                   /*offsetY*/ h + H - h, // !! OPRAVA !! Začátek posledních h řádků aktivní oblasti
                   /*sizeX*/ W - 2 * h, // Šířka bez levého a pravého rohu
                   /*sizeY*/ h, // Výška okraje
                   stride);
    }

    // Levá hrana (pro odeslání doleva)
    if (neighbours[LEFT] != MPI_PROC_NULL) {
        // Počítáme sloupce h až 2h-1, řádky 2h až h+H-h-1
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h,
                   /*offsetY*/ h + h,
                   /*sizeX*/ h, // Šířka okraje
                   /*sizeY*/ H - 2 * h, // Výška bez horního a dolního rohu
                   stride);
    }

    // Pravá hrana (pro odeslání doprava)
    if (neighbours[RIGHT] != MPI_PROC_NULL) {
        // Počítáme sloupce h+W-h až h+W-1, řádky 2h až h+H-h-1
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + W - h, // !! OPRAVA !! Začátek posledních h sloupců aktivní oblasti
                   /*offsetY*/ h + h,
                   /*sizeX*/ h, // Šířka okraje
                   /*sizeY*/ H - 2 * h, // Výška bez horního a dolního rohu
                   stride);
    }

    // --- Výpočet rohů ---

    // Levý horní roh
    if (neighbours[LEFT] != MPI_PROC_NULL && neighbours[TOP] != MPI_PROC_NULL) {
        // Počítáme oblast [h..2h-1][h..2h-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h,
                   /*offsetY*/ h,
                   /*sizeX*/ h,
                   /*sizeY*/ h,
                   stride);
    }
    // Pravý horní roh
    if (neighbours[RIGHT] != MPI_PROC_NULL && neighbours[TOP] != MPI_PROC_NULL) {
        // Počítáme oblast [h+W-h..h+W-1][h..2h-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + W - h, // !! OPRAVA !!
                   /*offsetY*/ h,
                   /*sizeX*/ h,
                   /*sizeY*/ h,
                   stride);
    }
    // Pravý dolní roh
    if (neighbours[RIGHT] != MPI_PROC_NULL && neighbours[BOTTOM] != MPI_PROC_NULL) {
        // Počítáme oblast [h+W-h..h+W-1][h+H-h..h+H-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + W - h, // !! OPRAVA !!
                   /*offsetY*/ h + H - h, // !! OPRAVA !!
                   /*sizeX*/ h,
                   /*sizeY*/ h,
                   stride);
    }
    // Levý dolní roh
    if (neighbours[LEFT] != MPI_PROC_NULL && neighbours[BOTTOM] != MPI_PROC_NULL) {
        // Počítáme oblast [h..2h-1][h+H-h..h+H-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h,
                   /*offsetY*/ h + H - h, // !! OPRAVA !!
                   /*sizeX*/ h,
                   /*sizeY*/ h,
                   stride);
    }
}

void ParallelHeatSolver::startHaloExchangeP2P(float* localData, std::array<MPI_Request, 8>& requests) {
    /**********************************************************************************************************************/
    /*                       Start the non-blocking halo zones exchange using P2P communication.                          */
    /*                         Use the requests array to return the requests from the function.                           */
    /*                            Don't forget to set the empty requests to MPI_REQUEST_NULL.                             */
    /**********************************************************************************************************************/

    requests.fill(MPI_REQUEST_NULL); // Inicializace pole požadavků na NULL

    int local_active_width = mLocalTileSize[0];
    int local_active_height = mLocalTileSize[1];
    int haloSize = haloZoneSize;
    int localBufferWidth = local_active_width + 2 * haloSize; // Celková šířka lokálního bufferu

    int send_tag = 10;
    int recv_tag = 10;


    // --- 1. Komunikace s HORNÍM sousedem (mTopRank) ---
    float* sendPtrTop = localData + haloSize * localBufferWidth + haloSize; // [h][h] - OK
    float* recvPtrTop = localData + haloSize; // [0][h] - OK

    MPI_Irecv(recvPtrTop, 1, horizontal_strip_type, mTopRank, recv_tag, gridComm, &requests[0]);
    MPI_Isend(sendPtrTop, 1, horizontal_strip_type, mTopRank, send_tag, gridComm, &requests[1]);

    // --- 2. Komunikace s DOLNÍM sousedem (mBottomRank) ---
    // Ukazatel na začátek PRVNÍHO řádku DOLNÍHO okrajového bloku (výška haloSize) v aktivní oblasti
    // Řádek má index: haloSize + local_active_height - haloSize
    float* sendPtrBottom =
        localData + (local_active_height) * localBufferWidth + haloSize; // [h+H-h][h]
    // ===============================
    float* recvPtrBottom = localData + (haloSize + local_active_height) * localBufferWidth + haloSize; // [h+H][h] - OK

    MPI_Irecv(recvPtrBottom, 1, horizontal_strip_type, mBottomRank, recv_tag, gridComm, &requests[2]);
    MPI_Isend(sendPtrBottom, 1, horizontal_strip_type, mBottomRank, send_tag, gridComm, &requests[3]);

    // --- 3. Komunikace s LEVÝM sousedem (mLeftRank) ---
    float* sendPtrLeft = localData + haloSize * localBufferWidth + haloSize; // [h][h] - OK
    float* recvPtrLeft = localData + haloSize * localBufferWidth; // [h][0] - OK

    MPI_Irecv(recvPtrLeft, 1, vertical_strip_type, mLeftRank, recv_tag, gridComm, &requests[4]);
    MPI_Isend(sendPtrLeft, 1, vertical_strip_type, mLeftRank, send_tag, gridComm, &requests[5]);

    // --- 4. Komunikace s PRAVÝM sousedem (mRightRank) ---
    // ========= OPRAVA ZDE ==========
    // Ukazatel na začátek PRVNÍHO sloupce PRAVÉHO okrajového bloku (šířka haloSize) v aktivní oblasti
    // Sloupec má index: haloSize + local_active_width - haloSize
    float* sendPtrRight =
        localData + haloSize * localBufferWidth + (haloSize + local_active_width - haloSize); // [h][h+W-h]
    // ===============================
    float* recvPtrRight = localData + haloSize * localBufferWidth + (haloSize + local_active_width); // [h][h+W] - OK

    MPI_Irecv(recvPtrRight, 1, vertical_strip_type, mRightRank, recv_tag, gridComm, &requests[6]);
    MPI_Isend(sendPtrRight, 1, vertical_strip_type, mRightRank, send_tag, gridComm, &requests[7]);
}

void ParallelHeatSolver::startHaloExchangeRMA(float* localData, MPI_Win window) {
    /**********************************************************************************************************************/
    /*                       Start the non-blocking halo zones exchange using RMA communication.                          */
    /*                   Do not forget that you put/get the values to/from the target's opposite side                     */
    /**********************************************************************************************************************/

    int local_active_width = mLocalTileSize[0];
    int local_active_height = mLocalTileSize[1];
    int haloSize = haloZoneSize;
    int localBufferWidth = local_active_width + 2 * haloSize; // Celková šířka lokálního bufferu

    MPI_Aint lbw = (MPI_Aint)localBufferWidth;
    MPI_Aint h = (MPI_Aint)haloSize;
    MPI_Aint H = (MPI_Aint)local_active_height;
    MPI_Aint W = (MPI_Aint)local_active_width;

    // --- Komunikace s LEVÝM sousedem (mLeftRank) ---
    if (mLeftRank != MPI_PROC_NULL) {
        float* origin_addr = localData + h * lbw + h; // [h][h] - OK
        MPI_Aint target_disp = h * lbw + h + W; // [h][h+W] - OK
        MPI_Put(origin_addr, 1, vertical_strip_type, mLeftRank, target_disp, 1, vertical_strip_type, window);
    }

    // --- Komunikace s PRAVÝM sousedem (mRightRank) ---
    if (mRightRank != MPI_PROC_NULL) {
        // ========= OPRAVA ZDE ==========
        // Ukazatel na začátek PRAVÉHO okrajového bloku [h][h+W-h]
        float* origin_addr = localData + h * lbw + (h + W - h);
        // ===============================
        MPI_Aint target_disp = h * lbw; // [h][0] - OK
        MPI_Put(origin_addr, 1, vertical_strip_type, mRightRank, target_disp, 1, vertical_strip_type, window);
    }

    // --- Komunikace s HORNÍM sousedem (mTopRank) ---
    if (mTopRank != MPI_PROC_NULL) {
        float* origin_addr = localData + h * lbw + h; // [h][h] - OK
        MPI_Aint target_disp = (h + H) * lbw + h; // [h+H][h] - OK
        MPI_Put(origin_addr, 1, horizontal_strip_type, mTopRank, target_disp, 1, horizontal_strip_type, window);
    }

    // --- Komunikace s DOLNÍM sousedem (mBottomRank) ---
    if (mBottomRank != MPI_PROC_NULL) {
        // ========= OPRAVA ZDE ==========
        // Ukazatel na začátek DOLNÍHO okrajového bloku [h+H-h][h]
        float* origin_addr = localData + (h + H - h) * lbw + h;
        // ===============================
        MPI_Aint target_disp = h * lbw +
            h; // [0][h] - Opraveno: Cíl je HORNÍ halo, začíná na řádku 0, sloupci h. Posun je h*lbw+h? Ne, jen h. Chyba!
        // Správný target_disp pro HORNÍ halo [0][h] je:
        target_disp = h; // Posun v počtu floatů od začátku bufferu
        // ===============================
        MPI_Put(origin_addr, 1, horizontal_strip_type, mBottomRank, target_disp, 1, horizontal_strip_type, window);
    }
}

void ParallelHeatSolver::awaitHaloExchangeP2P(std::array<MPI_Request, 8>& requests) {
    /**********************************************************************************************************************/
    /*                       Wait for all halo zone exchanges to finalize using P2P communication.                        */
    /**********************************************************************************************************************/

    std::array<MPI_Status, 8> status{};
    MPI_Waitall(requests.size(), &requests[0], &status[0]);
}

void ParallelHeatSolver::awaitHaloExchangeRMA(MPI_Win window) {
    /**********************************************************************************************************************/
    /*                       Wait for all halo zone exchanges to finalize using RMA communication.                        */
    /**********************************************************************************************************************/
    MPI_Win_fence(0, window);
}

void ParallelHeatSolver::printLocalTilesWithoutHalo() {
    std::stringstream ss; // Vytvoříme stringstream pro sestavení výstupu

    ss << "\nRank: " << mWorldRank << " - Aktivní oblast dlaždice (mLocalTileTemperature[0]); " << mLocalTileSize[0] <<
        "x" << mLocalTileSize[1] << "\n";
    const int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;

    // Nastavíme formátování pro čísla s plovoucí desetinnou čárkou (volitelné)
    ss << std::fixed << std::setprecision(1); // Např. 4 desetinná místa

    // Iterujeme POUZE přes aktivní oblast dlaždice
    for (int y = 0; y < mLocalTileSize[1]; ++y) {
        // Vnější smyčka přes řádky (y) aktivní oblasti
        for (int x = 0; x < mLocalTileSize[0]; ++x) {
            size_t index = static_cast<size_t>(y + haloZoneSize) * localBufferWidth
                + static_cast<size_t>(x + haloZoneSize);

            // Přidáme hodnotu z aktivní oblasti
            ss << mLocalTileTemperature[0].at(index);

            ss << " ";
        }
        // Na konci každého řádku aktivní oblasti přidáme nový řádek
        ss << "\n";
    }

    std::cout << ss.str();
}

void ParallelHeatSolver::printLocalTilesWithHalo() {
    std::stringstream ss; // Vytvoříme stringstream pro sestavení výstupu

    // Vypočítáme celkové rozměry bufferu
    const int totalHeight = mLocalTileSize[1] + 2 * haloZoneSize;
    const int totalWidth = mLocalTileSize[0] + 2 * haloZoneSize;
    const int localBufferWidth = totalWidth; // Stride je celková šířka

    // Přidáme hlavičku informující o tisku celého bufferu
    ss << "\nRank: " << mWorldRank << " - Celý lokální buffer (mLocalTileTemperature[0]); "
        << totalWidth << "x" << totalHeight << " (včetně halo)\n";

    // Nastavíme formátování (můžete si upravit přesnost)
    ss << std::fixed << std::setprecision(1);

    // Iterujeme přes CELÝ buffer, včetně halo zón
    for (int y = 0; y < totalHeight; ++y) {
        // Vnější smyčka přes všechny řádky bufferu
        for (int x = 0; x < totalWidth; ++x) {
            // Vnitřní smyčka přes všechny sloupce bufferu
            // Index je nyní přímý, protože x a y jsou souřadnice v bufferu
            size_t index = static_cast<size_t>(y) * localBufferWidth + static_cast<size_t>(x);

            // Přidáme hodnotu z bufferu
            // Použijeme .at() pro bezpečnost, nebo [] pro rychlost

            ss << mLocalTileTemperature[0].at(index);


            // Přidáme oddělovač pro čitelnost
            ss << " ";
        }
        // Na konci každého řádku bufferu přidáme nový řádek
        ss << "\n";
    }

    // Vytiskneme celý sestavený string najednou
    std::cout << ss.str();
}

void ParallelHeatSolver::printGlobalGridAligned() const {
    // Tisk provádí pouze root proces
    if (mWorldRank != ROOT) {
        return;
    }
    auto globalData = mMaterialProps.getInitialTemperature().data();

    // Zkontrolujeme, zda máme platná data
    if (globalData == nullptr) {
        std::fprintf(stderr, "[Rank %d] Chyba: printGlobalGridAligned voláno s nullptr!\n", mWorldRank);
        return;
    }

    const size_t edgeSize = mMaterialProps.getEdgeSize();
    if (edgeSize == 0) {
        std::printf("[Rank %d] Globální mřížka má velikost 0.\n", mWorldRank);
        return;
    }

    std::stringstream ss;

    ss << "\n[Rank " << mWorldRank << "] Globální mřížka teplot ("
        << edgeSize << "x" << edgeSize << ", zarovnáno):\n";

    // --- Nastavení formátování a šířky ---
    const int precision = 1; // Počet desetinných míst
    const int fieldWidth = 8; // Pevná šířka pole pro každé číslo

    ss << std::fixed << std::setprecision(precision);
    // ------------------------------------

    // Iterujeme přes globální mřížku
    for (size_t y = 0; y < edgeSize; ++y) {
        // Vnější smyčka přes řádky
        for (size_t x = 0; x < edgeSize; ++x) {
            // Vnitřní smyčka přes sloupce
            // Index v lineárně uloženém globálním poli
            size_t index = y * edgeSize + x;

            // Nastavíme šířku pole PŘED vložením čísla
            ss << std::setw(fieldWidth);
            ss << globalData[index]; // Přidáme hodnotu z globálního pole

            // Mezeru už nepotřebujeme, setw zajistí oddělení
        }
        ss << "\n"; // Nový řádek po dokončení řádku mřížky
    }

    // Vytiskneme celý sestavený a zarovnaný string najednou
    std::cout << ss.str();
}

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>>& outResult) {
    std::array<MPI_Request, 8> requestsP2P{};
    std::array<MPI_Request, 8> paramsRequests{};

    /**********************************************************************************************************************/
    /*                                         Scatter initial data.                                                      */
    /**********************************************************************************************************************/

    const float* globalTemperatures = nullptr;
    const float* globalDomainParameters = nullptr;
    const int* globalDomainMap = nullptr;
    if (rank(gridComm) == ROOT) {
        globalTemperatures = mMaterialProps.getInitialTemperature().data();
        globalDomainParameters = mMaterialProps.getDomainParameters().data();
        globalDomainMap = mMaterialProps.getDomainMap().data();
    }


    // Scatterujeme do 'newIdx', protože to bude výsledek "iterace -1"
    float* localTemperatureBuffer = mLocalTileTemperature[0].data();
    // Předpoklad: mLocalTileMaterialProp je pro float parametry
    float* localMaterialPropBuffer = mLocalTileMaterialProp.data();
    // Předpoklad: mLocalTileMaterial je pro int mapu
    int* localMaterialMapBuffer = mLocalTileMaterial.data();

    localDomainMap<float>(globalTemperatures, localTemperatureBuffer);

    // Vlastnosti materiálu (parametry - float)
    localDomainMap<float>(globalDomainParameters, localMaterialPropBuffer);

    // Mapa materiálu (int)
    localDomainMap<int>(globalDomainMap, localMaterialMapBuffer);

    /**********************************************************************************************************************/
    /* Exchange halo zones of initial domain temperature and parameters using P2P communication. Wait for them to finish. */
    /**********************************************************************************************************************/

    startHaloExchangeP2P(localTemperatureBuffer, requestsP2P);
    startHaloExchangeP2P(localMaterialPropBuffer, paramsRequests);
    awaitHaloExchangeP2P(requestsP2P);
    awaitHaloExchangeP2P(paramsRequests);


    /**********************************************************************************************************************/
    /*                            Copy initial temperature to the second buffer.                                          */
    /**********************************************************************************************************************/

    std::copy(mLocalTileTemperature[0].begin(), // Zdroj je neinicializovaný buffer 1
              mLocalTileTemperature[0].end(),
              mLocalTileTemperature[1].begin());


    double startTime = MPI_Wtime();

    // 3. Start main iterative simulation loop.
    for (std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter) {
        const std::size_t oldIdx = iter % 2; // Index of the buffer with old temperatures
        const std::size_t newIdx = (iter + 1) % 2; // Index of the buffer with new temperatures

        /**********************************************************************************************************************/
        /*                            Compute and exchange halo zones using P2P or RMA.                                       */
        /**********************************************************************************************************************/

        computeHaloZones(mLocalTileTemperature[oldIdx].data(), mLocalTileTemperature[newIdx].data());

        if (mSimulationProps.isRunParallelP2P()) {
            // Start halo exchange
            startHaloExchangeP2P(mLocalTileTemperature[newIdx].data(), requestsP2P);
        }
        else if (mSimulationProps.isRunParallelRMA()) {
            // Start halo exchange
            MPI_Win_fence(0, m_rma_win[newIdx]);
            startHaloExchangeRMA(mLocalTileTemperature[newIdx].data(), m_rma_win[newIdx]);
        }


        /**********************************************************************************************************************/
        /*                           Compute the rest of the tile. Use updateTile method.                                     */
        /**********************************************************************************************************************/

        // Získání rozměrů čisté aktivní oblasti
        const int local_active_width = mLocalTileSize[0];
        const int local_active_height = mLocalTileSize[1];
        const int h_size = haloZoneSize;

        // Výpočet rozměrů a offsetů pro VNITŘNÍ oblast
        const std::size_t inner_offset_x = static_cast<std::size_t>(h_size +
            h_size); // Začíná za horním a levým okrajem = 2*h
        const std::size_t inner_offset_y = static_cast<std::size_t>(h_size +
            h_size); // Začíná za horním a levým okrajem = 2*h
        const std::size_t inner_size_x = static_cast<std::size_t>(local_active_width -
            2 * h_size); // Šířka bez levého a pravého okraje
        const std::size_t inner_size_y = static_cast<std::size_t>(local_active_height -
            2 * h_size); // Výška bez horního a dolního okraje
        const std::size_t stride = static_cast<std::size_t>(local_active_width + 2 * h_size); // Celková šířka bufferu

        // Voláme updateTile POUZE pro vnitřní oblast, pokud existuje
        if (inner_size_x > 0 && inner_size_y > 0) {
            updateTile(mLocalTileTemperature[oldIdx].data(),
                       mLocalTileTemperature[newIdx].data(),
                       mLocalTileMaterialProp.data(),
                       mLocalTileMaterial.data(),
                       inner_offset_x, // offsetX
                       inner_offset_y, // offsetY
                       inner_size_x, // sizeX
                       inner_size_y, // sizeY
                       stride); // stride
        }


        /**********************************************************************************************************************/
        /*                            Wait for all halo zone exchanges to finalize.                                           */
        /**********************************************************************************************************************/

        if (!mSimulationProps.isRunParallelRMA()) {
            // Wait for halo exchange
            awaitHaloExchangeP2P(requestsP2P);
            // printLocalTilesWithHalo();
        }
        else {
            // Wait for halo exchange
            awaitHaloExchangeRMA(m_rma_win[newIdx]);
            // printLocalTilesWithHalo();
        }


        if (shouldStoreData(iter)) {
            /**********************************************************************************************************************/
            /*                          Store the data into the output file using parallel or sequential IO.                      */
            /**********************************************************************************************************************/

            if (mSimulationProps.useParallelIO()) {
                storeDataIntoFileParallel(mFileHandle, iter, mLocalTileTemperature[newIdx].data());
            }
            else {
                gatherTiles<float>(mLocalTileTemperature[newIdx].data(), outResult.data());

                storeDataIntoFileSequential(mFileHandle, iter, outResult.data());
            }
        }

        if (shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature()) {
            /**********************************************************************************************************************/
            /*                 Compute and print middle column average temperature and print progress report.                     */
            /**********************************************************************************************************************/
            float middleColAvgTemp = computeMiddleColumnAverageTemperatureParallel(
                mLocalTileTemperature[newIdx].data());

            int rank;
            MPI_Comm_rank(avgTempComm, &rank);

            if (rank == 0)
                printProgressReport(iter, middleColAvgTemp);
        }
    }

    const std::size_t resIdx = mSimulationProps.getNumIterations() % 2; // Index of the buffer with final temperatures

    double elapsedTime = MPI_Wtime() - startTime;

    /**********************************************************************************************************************/
    /*                                     Gather final domain temperature.                                               */
    /**********************************************************************************************************************/

    gatherTiles<float>(mLocalTileTemperature[resIdx].data(), outResult.data());

    /**********************************************************************************************************************/
    /*           Compute (sequentially) and report final middle column temperature average and print final report.        */
    /**********************************************************************************************************************/

    if (mWorldRank == 0) {
        auto avg = computeMiddleColumnAverageTemperatureSequential(outResult.data());
        printFinalReport(elapsedTime, avg);
    }
}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const {
    /**********************************************************************************************************************/
    /*                Return true if rank should compute middle column average temperature.                               */
    /**********************************************************************************************************************/

    return (avgTempComm != MPI_COMM_NULL);
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureParallel(const float* localData) const {
    /**********************************************************************************************************************/
    /*                  Implement parallel middle column average temperature computation.                                 */
    /*                      Use OpenMP directives to accelerate the local computations.                                   */
    /**********************************************************************************************************************/


    float local_sum = 0.0;

    int nRanksInComm;
    MPI_Comm_size(avgTempComm, &nRanksInComm);

    const int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;

    // Použití OpenMP pro paralelizaci lokálního sčítání
#pragma omp parallel for reduction(+:local_sum) schedule(static)
    for (std::size_t y = haloZoneSize; y < mLocalTileSize[1] + haloZoneSize; ++y) {
        local_sum += localData[y * localBufferWidth + haloZoneSize + mLocalTileSize[0] / 2];
    }


    if (nRanksInComm == 1) {
        return local_sum / static_cast<float>(mLocalTileSize[1]);
    }

    float global_sum = 0.0;

    MPI_Reduce(
        &local_sum, // Adresa lokální hodnoty k odeslání
        &global_sum, // Adresa, kam se uloží výsledek (globální suma)
        1, // Počet prvků (posíláme jednu sumu)
        MPI_FLOAT, // Datový typ sčítaných hodnot (používáme double)
        MPI_SUM, // Operace, kterou chceme provést (sčítání),
        0,
        avgTempComm // Komunikátor obsahující JEN procesy ve středním sloupci!
    );


    return global_sum / static_cast<float>(mLocalTileSize[1] * nRanksInComm);
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureSequential(const float* globalData) const {
    /**********************************************************************************************************************/
    /*                  Implement sequential middle column average temperature computation.                               */
    /*                      Use OpenMP directives to accelerate the local computations.                                   */
    /**********************************************************************************************************************/


    // Zkontroluj, zda máme platná data (mělo by být voláno jen na rootu!)
    if (globalData == nullptr) {
        // Můžeš vrátit nějakou chybovou hodnotu nebo vyhodit výjimku,
        // ale ideálně by se sem kód na ne-root procesu neměl dostat.
        // fprintf(stderr, "ERROR: computeMiddleColumnAverageTemperatureSequential called with nullptr!\n");
        return -1.0f; // Nebo jiná indikace chyby
    }

    double middleColSum = 0.0; // Použij double pro sumu pro lepší přesnost
    const std::size_t edgeSize = mMaterialProps.getEdgeSize();

    // Zkontroluj, zda edgeSize je > 0, aby se předešlo dělení nulou
    if (edgeSize == 0) {
        return 0.0f; // Prázdná mřížka
    }

    const std::size_t middleColIndex = edgeSize / 2; // Index prostředního sloupce

    // POZOR: OpenMP zde dává smysl, pokud je edgeSize velké a funkce je volána jen na rootu.
#pragma omp parallel for reduction(+:middleColSum) schedule(static)
    for (std::size_t y = 0; y < edgeSize; ++y) {
        // Iterujeme přes řádky
        // Spočítáme index bodu v prostředním sloupci pro daný řádek y
        std::size_t index = y * edgeSize + middleColIndex;
        middleColSum += globalData[index];
    }

    // Vypočítáme průměr dělením počtem bodů ve sloupci (což je edgeSize)
    return static_cast<float>(middleColSum / edgeSize);
}

void ParallelHeatSolver::openOutputFileSequential() {
    // Create the output file for sequential access.
    mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                            H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (!mFileHandle.valid()) {
        throw std::ios::failure("Cannot create output file!");
    }
}

void ParallelHeatSolver::storeDataIntoFileSequential(hid_t fileHandle,
                                                     std::size_t iteration,
                                                     const float* globalData) {
    storeDataIntoFile(fileHandle, iteration, globalData);
}

void ParallelHeatSolver::openOutputFileParallel() {
#ifdef H5_HAVE_PARALLEL
    Hdf5PropertyListHandle faplHandle{};

    /**********************************************************************************************************************/
    /*                          Open output HDF5 file for parallel access with alignment.                                 */
    /*      Set up faplHandle to use MPI-IO and alignment. The handle will automatically release the resource.            */
    /**********************************************************************************************************************/


    faplHandle = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(faplHandle, gridComm, MPI_INFO_NULL);

    // alignment for file access
    hsize_t alignment = 1024 * 1024;
    H5Pset_alignment(faplHandle, 0, alignment);

    mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                            H5F_ACC_TRUNC,
                            H5P_DEFAULT,
                            faplHandle);

    if (!mFileHandle.valid()) {
        throw std::ios::failure("Cannot create output file!");
    }
#else
    throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}

void ParallelHeatSolver::storeDataIntoFileParallel(hid_t fileHandle,
                                                   [[maybe_unused]] std::size_t iteration,
                                                   [[maybe_unused]] const float* localData) {
    if (fileHandle == H5I_INVALID_HID) {
        return;
    }

#ifdef H5_HAVE_PARALLEL
    std::array gridSize{
        static_cast<hsize_t>(mMaterialProps.getEdgeSize()),
        static_cast<hsize_t>(mMaterialProps.getEdgeSize())
    };

    // Create new HDF5 group in the output file
    std::string groupName = "Timestep_" + std::to_string(iteration / mSimulationProps.getWriteIntensity());

    Hdf5GroupHandle groupHandle(H5Gcreate(fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    {
        /**********************************************************************************************************************/
        /*                                Compute the tile offsets and sizes.                                                 */
        /*               Note that the X and Y coordinates are swapped (but data not altered).                                */
        /**********************************************************************************************************************/
        std::array<hsize_t, 2> tileSize{
            static_cast<hsize_t>(mLocalTileSize[0]), static_cast<hsize_t>(mLocalTileSize[1])
        };
        std::array<hsize_t, 2> localDataTileOffset{haloZoneSize, haloZoneSize};

        hsize_t offsetX{0};
        hsize_t offsetY{0};

        if (myCoorsGrid[0] == 1) {
            // 1d
            offsetX = myCoorsGrid[1] * mLocalTileSize[0];
        }
        else if (myCoorsGrid[1] == 1) {
            // 1d
            offsetY = myCoorsGrid[1] * mLocalTileSize[0];
        }
        else {
            offsetX = myCoorsGrid[1] * mLocalTileSize[0];
            offsetY = myCoorsGrid[0] * mLocalTileSize[1];
        }

        std::array<hsize_t, 2> tileOffsetInGlobal{offsetY, offsetX};

        // Create new dataspace and dataset using it.
        static constexpr std::string_view dataSetName{"Temperature"};


        Hdf5PropertyListHandle datasetPropListHandle{};
        /**********************************************************************************************************************/
        /*                            Create dataset property list to set up chunking.                                        */
        /*                Set up chunking for collective write operation in datasetPropListHandle variable.                   */
        /**********************************************************************************************************************/
        datasetPropListHandle = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(datasetPropListHandle, 2, tileSize.data());

        Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));
        Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                                  H5T_NATIVE_FLOAT, dataSpaceHandle,
                                                  H5P_DEFAULT, datasetPropListHandle,
                                                  H5P_DEFAULT));

        Hdf5DataspaceHandle memSpaceHandle{};
        /**********************************************************************************************************************/
        /*                Create memory dataspace representing tile in the memory (set up memSpaceHandle).                    */
        /**********************************************************************************************************************/
        std::array<hsize_t, 2> tileSizeWithHaloZones{
            mLocalTileSize[1] + 2 * haloZoneSize, mLocalTileSize[0] + 2 * haloZoneSize
        };
        memSpaceHandle = H5Screate_simple(2, tileSizeWithHaloZones.data(), nullptr);

        /**********************************************************************************************************************/
        /*              Select inner part of the tile in memory and matching part of the dataset in the file                  */
        /*                           (given by position of the tile in global domain).                                        */
        /**********************************************************************************************************************/
        H5Sselect_hyperslab(memSpaceHandle, H5S_SELECT_SET, localDataTileOffset.data(), nullptr, tileSize.data(),
                            nullptr);
        H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, tileOffsetInGlobal.data(), nullptr, tileSize.data(),
                            nullptr);

        Hdf5PropertyListHandle propListHandle{};

        /**********************************************************************************************************************/
        /*              Perform collective write operation, writting tiles from all processes at once.                        */
        /*                                   Set up the propListHandle variable.                                              */
        /**********************************************************************************************************************/
        // create XFER property list and set collective IO
        propListHandle = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(propListHandle, H5FD_MPIO_COLLECTIVE);

        // write
        H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, dataSpaceHandle, propListHandle, localData);
    }

    {
        // 3. Store attribute with current iteration number in the group.
        static constexpr std::string_view attributeName{"Time"};
        Hdf5DataspaceHandle dataSpaceHandle(H5Screate(H5S_SCALAR));
        Hdf5AttributeHandle attributeHandle(H5Acreate2(groupHandle, attributeName.data(),
                                                       H5T_IEEE_F64LE, dataSpaceHandle,
                                                       H5P_DEFAULT, H5P_DEFAULT));
        const double snapshotTime = static_cast<double>(iteration);
        H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
    }
#else
    throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}
