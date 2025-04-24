/**
 * @file    ParallelHeatSolver.cpp
 *
 * @author  Jakub Vlk <xvlkja07@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-02-23
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <ios>
#include <string_view>
#include <sstream>
#include <iomanip>

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
                mFileHandle = H5Fcreate(simulationProps.getOutputFileName(codeType).c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
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

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &gridComm);
    MPI_Comm_set_name(gridComm, "GridComm");

    int my_grid_rank;
    MPI_Comm_rank(gridComm, &my_grid_rank);

    MPI_Cart_coords(gridComm, my_grid_rank, ndims, myCoorsGrid);

    MPI_Comm_size(gridComm, &m_size);

    int middle_col_index = dims[0] / 2;
    int color = (myCoorsGrid[0] == middle_col_index) ? 1 : MPI_UNDEFINED;

    int key = myCoorsGrid[1];

    // Split the grid communicator to create a communicator for the middle column
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

    mLocalTileSize[0] = mGridSizeEdge / nX; // Local tile width (X dimension)
    mLocalTileSize[1] = mGridSizeEdge / nY; // Local tile height (Y dimension)

    int local_tile_width = mLocalTileSize[0];
    int local_tile_height = mLocalTileSize[1];
    int global_grid_width = mMaterialProps.getEdgeSize();

    // MPI Datatype representing the layout of a tile within the global grid (for Scatterv/Gatherv on root)
    MPI_Datatype tile_layout_type_float;
    MPI_Type_vector(local_tile_height, local_tile_width, global_grid_width, MPI_FLOAT, &tile_layout_type_float);

    MPI_Aint float_lb, float_extent;
    MPI_Type_get_extent(MPI_FLOAT, &float_lb, &float_extent);
    MPI_Type_create_resized(tile_layout_type_float, 0, float_extent, &tileTypeFloat);
    MPI_Type_free(&tile_layout_type_float);
    MPI_Type_commit(&tileTypeFloat);

    // Same for INT
    MPI_Datatype tile_layout_type_int;
    MPI_Type_vector(local_tile_height, local_tile_width, global_grid_width, MPI_INT, &tile_layout_type_int);

    MPI_Aint int_lb, int_extent;
    MPI_Type_get_extent(MPI_INT, &int_lb, &int_extent);
    MPI_Type_create_resized(tile_layout_type_int, 0, int_extent, &tileTypeInt);
    MPI_Type_free(&tile_layout_type_int);
    MPI_Type_commit(&tileTypeInt);

    int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;

    // MPI Datatype representing the layout of the active area within the local buffer (for Scatterv/Gatherv on non-root ranks)
    MPI_Type_vector(mLocalTileSize[1], mLocalTileSize[0], localBufferWidth, MPI_FLOAT,
                    &m_localActiveAreaTypeFloat);
    MPI_Type_commit(&m_localActiveAreaTypeFloat);

    // Same for INT
    MPI_Type_vector(mLocalTileSize[1], mLocalTileSize[0], localBufferWidth, MPI_INT,
                    &m_localActiveAreaTypeInt);
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
    // std::vector automatically manages memory
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

    // MPI Datatype for a horizontal strip (row(s)) of data within the local buffer
    MPI_Type_vector(
        haloSize, // Number of blocks (rows)
        local_active_width, // Elements per block (columns)
        local_buffer_stride, // Stride in elements (total columns in buffer)
        MPI_FLOAT,
        &horizontal_strip_type
    );
    MPI_Type_commit(&horizontal_strip_type);

    // MPI Datatype for a vertical strip (column(s)) of data within the local buffer
    MPI_Type_vector(
        local_active_height, // Number of blocks (rows)
        haloSize, // Elements per block (columns)
        local_buffer_stride, // Stride in elements (total columns in buffer)
        MPI_FLOAT,
        &vertical_strip_type
    );
    MPI_Type_commit(&vertical_strip_type);

    // Get ranks of neighbors in the Cartesian grid
    MPI_Cart_shift(
        gridComm,
        0, // Shift along X axis
        1, // Shift by +1 (right)
        &mLeftRank, // Rank shifted by -1 (left)
        &mRightRank // Rank shifted by +1 (right)
    );
    MPI_Cart_shift(
        gridComm,
        1, // Shift along Y axis
        1, // Shift by +1 (down)
        &mTopRank, // Rank shifted by -1 (up)
        &mBottomRank // Rank shifted by +1 (down)
    );

    // Create MPI Windows for RMA if enabled
    if (mSimulationProps.isRunParallelRMA()) {
        // Calculate window size based on the total size of the local buffer (including halo)
        MPI_Aint window_size_bytes = (MPI_Aint)(local_active_width + 2 * haloZoneSize)
            * (MPI_Aint)(local_active_height + 2 * haloZoneSize)
            * sizeof(float);

        // Create window for the first temperature buffer
        float* window_base_ptr = mLocalTileTemperature[0].data();
        MPI_Win_create(
            window_base_ptr,
            window_size_bytes,
            sizeof(float), // Displacement unit is size of one float
            MPI_INFO_NULL,
            gridComm, // Window is collective over the grid communicator
            &m_rma_win[0]
        );

        // Create window for the second temperature buffer
        window_base_ptr = mLocalTileTemperature[1].data();
        MPI_Win_create(
            window_base_ptr,
            window_size_bytes,
            sizeof(float), // Displacement unit is size of one float
            MPI_INFO_NULL,
            gridComm, // Window is collective over the grid communicator
            &m_rma_win[1]
        );
    }
}

void ParallelHeatSolver::deinitHaloExchange() {
    /**********************************************************************************************************************/
    /*                            Deinitialize variables and MPI datatypes for halo exchange.                             */
    /**********************************************************************************************************************/
    MPI_Type_free(&horizontal_strip_type);
    MPI_Type_free(&vertical_strip_type);

    if (m_rma_win[0] != MPI_WIN_NULL) {
        MPI_Win_free(&m_rma_win[0]);
    }
    if (m_rma_win[1] != MPI_WIN_NULL) {
        MPI_Win_free(&m_rma_win[1]);
    }
}

template <typename T>
void ParallelHeatSolver::localDomainMap(const T* globalData, T* localData) {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported scatter datatype!");

    /**********************************************************************************************************************/
    /*                      Implement master's global tile scatter to each rank's local tile.                             */
    /**********************************************************************************************************************/
    const MPI_Datatype globalSendTileType = std::is_same_v<T, int> ? tileTypeInt : tileTypeFloat;
    const MPI_Datatype localRecvAreaType = std::is_same_v<T, int>
                                               ? m_localActiveAreaTypeInt
                                               : m_localActiveAreaTypeFloat;

    std::vector<int> sendcounts; // Only used on root
    std::vector<int> displs; // Only used on root

    int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;
    // Calculate offset to the start of the active area within the local buffer
    int offset = haloZoneSize * localBufferWidth + haloZoneSize;
    T* activeAreaStartPtr = localData + offset; // Pointer to the start of the active area

    // Root process prepares sendcounts and displacements
    if (rank(gridComm) == ROOT) {
        sendcounts.resize(m_size);
        displs.resize(m_size);

        for (int i = 0; i < m_size; ++i) {
            // Each process receives one block described by localRecvAreaType
            sendcounts[i] = 1;

            // Calculate the displacement for process 'i' in the global grid
            int coords[2]; // Coordinates of process 'i' in the processor grid
            MPI_Cart_coords(gridComm, i, 2, coords);

            // Calculate start row and column in the global grid
            int startRow = coords[1] * mLocalTileSize[1]; // Y-coord * tile_height
            int startCol = coords[0] * mLocalTileSize[0]; // X-coord * tile_width

            int globalWidth = mMaterialProps.getEdgeSize();

            // Calculate linear displacement in terms of elements
            displs[i] = startRow * globalWidth + startCol;
        }
    }

    // Perform the Scatterv operation
    MPI_Scatterv(
        globalData, // sendbuf: Global buffer on root
        sendcounts.data(), // sendcounts: Array of counts (1 for each process)
        displs.data(), // displs: Array of displacements in global buffer
        globalSendTileType, // sendtype: Type describing the block layout in the global buffer
        activeAreaStartPtr, // recvbuf: Pointer to the start of the active area in the local buffer
        1, // recvcount: Receiving 1 block of type localRecvAreaType
        localRecvAreaType, // recvtype: Type describing the block layout in the local buffer
        ROOT, // Rank of the root process
        gridComm // Communicator for the grid
    );
}

template <typename T>
void ParallelHeatSolver::gatherTiles(const T* localData, T* globalData) {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported gather datatype!");

    /**********************************************************************************************************************/
    /*                      Implement each rank's local tile gather to master's rank global tile.                         */
    /**********************************************************************************************************************/
    const MPI_Datatype globalRecvTileType = std::is_same_v<T, int> ? tileTypeInt : tileTypeFloat;
    const MPI_Datatype localSendAreaType = std::is_same_v<T, int>
                                               ? m_localActiveAreaTypeInt
                                               : m_localActiveAreaTypeFloat;
    int globalWidth = mMaterialProps.getEdgeSize();

    std::vector<int> recvcounts; // Only used on root
    std::vector<int> displs; // Only used on root

    // Calculate pointer to the start of the active area to send
    int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;
    int offset = haloZoneSize * localBufferWidth + haloZoneSize;
    const T* activeAreaStartPtr = localData + offset; // Pointer to the start of the active area

    if (mWorldRank == ROOT) {
        recvcounts.resize(m_size);
        displs.resize(m_size);

        for (int i = 0; i < m_size; ++i) {
            // Root receives one block described by globalRecvTileType from each process
            recvcounts[i] = 1;

            // Calculate the displacement for process 'i' in the global buffer
            int coords[2];
            MPI_Cart_coords(gridComm, i, 2, coords);

            // Calculate start row and column
            int startRow = coords[1] * mLocalTileSize[1]; // Y * local tile height
            int startCol = coords[0] * mLocalTileSize[0]; // X * local tile width

            displs[i] = startRow * globalWidth + startCol; // Linear displacement in global buffer
        }
    }

    // Perform the Gatherv operation
    MPI_Gatherv(
        activeAreaStartPtr, // sendbuf: Pointer to the start of the active area to send
        1, // sendcount: Sending 1 block of type localSendAreaType
        localSendAreaType, // sendtype: Type describing the active area in the local buffer
        globalData, // recvbuf: Global buffer on root
        recvcounts.data(), // recvcounts: Array of counts (1) on root
        displs.data(), // displs: Array of displacements in the global buffer on root
        globalRecvTileType, // recvtype: Type describing the tile layout in the global buffer on root
        ROOT, // Rank of the root process
        gridComm // Communicator for the grid
    );
}

void ParallelHeatSolver::computeHaloZones(const float* oldTemp, float* newTemp) {
    /**********************************************************************************************************************/
    /*  Compute new temperatures in halo zones, so that copy operations can be overlapped with inner region computation.  */
    /*                        Use updateTile method to compute new temperatures in halo zones.                            */
    /*                             TAKE CARE NOT TO COMPUTE THE SAME AREAS TWICE                                          */
    /**********************************************************************************************************************/

    const int h = haloZoneSize;
    const int W = mLocalTileSize[0]; // Active area width
    const int H = mLocalTileSize[1]; // Active area height
    // Total buffer width (stride)
    const std::size_t stride = static_cast<std::size_t>(W + 2 * h);
    const float* params = mLocalTileMaterialProp.data();
    const int* map = mLocalTileMaterial.data();

    // Compute boundary regions of the *active* area that will be sent to neighbors.
    // These are computed into the *newTemp* buffer.

    // Top edge (excluding corners)
    if (mTopRank != MPI_PROC_NULL) {
        // Region: [h..h+h-1][h+h..h+W-h-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + h,
                   /*offsetY*/ h,
                   /*sizeX*/ W - 2 * h, // Width without left/right corners
                   /*sizeY*/ h, // Height of the edge
                   stride);
    }

    // Bottom edge (excluding corners)
    if (mBottomRank != MPI_PROC_NULL) {
        // Region: [h+H-h..h+H-1][h+h..h+W-h-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + h,
                   /*offsetY*/ h + H - h, // Starting row of the bottom edge
                   /*sizeX*/ W - 2 * h, // Width without left/right corners
                   /*sizeY*/ h, // Height of the edge
                   stride);
    }

    // Left edge (excluding corners)
    if (mLeftRank != MPI_PROC_NULL) {
        // Region: [h+h..h+H-h-1][h..h+h-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h,
                   /*offsetY*/ h + h,
                   /*sizeX*/ h, // Width of the edge
                   /*sizeY*/ H - 2 * h, // Height without top/bottom corners
                   stride);
    }

    // Right edge (excluding corners)
    if (mRightRank != MPI_PROC_NULL) {
        // Region: [h+h..h+H-h-1][h+W-h..h+W-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + W - h, // Starting column of the right edge
                   /*offsetY*/ h + h,
                   /*sizeX*/ h, // Width of the edge
                   /*sizeY*/ H - 2 * h, // Height without top/bottom corners
                   stride);
    }

    // --- Compute corners ---

    // Top-Left corner
    if (mLeftRank != MPI_PROC_NULL && mTopRank != MPI_PROC_NULL) {
        // Region: [h..h+h-1][h..h+h-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h,
                   /*offsetY*/ h,
                   /*sizeX*/ h,
                   /*sizeY*/ h,
                   stride);
    }
    // Top-Right corner
    if (mRightRank != MPI_PROC_NULL && mTopRank != MPI_PROC_NULL) {
        // Region: [h..h+h-1][h+W-h..h+W-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + W - h, // Starting column of the right corner
                   /*offsetY*/ h,
                   /*sizeX*/ h,
                   /*sizeY*/ h,
                   stride);
    }
    // Bottom-Right corner
    if (mRightRank != MPI_PROC_NULL && mBottomRank != MPI_PROC_NULL) {
        // Region: [h+H-h..h+H-1][h+W-h..h+W-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h + W - h, // Starting column of the right corner
                   /*offsetY*/ h + H - h, // Starting row of the bottom corner
                   /*sizeX*/ h,
                   /*sizeY*/ h,
                   stride);
    }
    // Bottom-Left corner
    if (mLeftRank != MPI_PROC_NULL && mBottomRank != MPI_PROC_NULL) {
        // Region: [h+H-h..h+H-1][h..h+h-1]
        updateTile(oldTemp, newTemp, params, map,
                   /*offsetX*/ h,
                   /*offsetY*/ h + H - h, // Starting row of the bottom corner
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

    requests.fill(MPI_REQUEST_NULL);

    int local_active_width = mLocalTileSize[0];
    int local_active_height = mLocalTileSize[1];
    int haloSize = haloZoneSize;
    int localBufferWidth = local_active_width + 2 * haloSize; // Total width of the local buffer

    int send_tag = 10;
    int recv_tag = 10;

    // --- 1. Communication with TOP neighbor (mTopRank) ---
    // Send: Top boundary of the active region [h..h+h-1][h..h+W-1]
    // Recv: Into the top halo region [0..h-1][h..h+W-1]
    float* sendPtrTop = localData + haloSize * localBufferWidth + haloSize; // Pointer to [h][h]
    float* recvPtrTop = localData + haloSize; // Pointer to [0][h] (start of top halo, skipping left halo)

    MPI_Irecv(recvPtrTop, 1, horizontal_strip_type, mTopRank, recv_tag, gridComm, &requests[0]);
    MPI_Isend(sendPtrTop, 1, horizontal_strip_type, mTopRank, send_tag, gridComm, &requests[1]);

    // --- 2. Communication with BOTTOM neighbor (mBottomRank) ---
    // Send: Bottom boundary of the active region [h+H-h..h+H-1][h..h+W-1]
    // Recv: Into the bottom halo region [h+H..h+H+h-1][h..h+W-1]
    float* sendPtrBottom =
        localData + (haloSize + local_active_height - haloSize) * localBufferWidth + haloSize;
    // Pointer to [h+H-h][h] (start of bottom active edge)
    float* recvPtrBottom = localData + (haloSize + local_active_height) * localBufferWidth + haloSize;
    // Pointer to [h+H][h] (start of bottom halo, skipping left halo)

    MPI_Irecv(recvPtrBottom, 1, horizontal_strip_type, mBottomRank, recv_tag, gridComm, &requests[2]);
    MPI_Isend(sendPtrBottom, 1, horizontal_strip_type, mBottomRank, send_tag, gridComm, &requests[3]);

    // --- 3. Communication with LEFT neighbor (mLeftRank) ---
    // Send: Left boundary of the active region [h..h+H-1][h..h+h-1]
    // Recv: Into the left halo region [h..h+H-1][0..h-1]
    float* sendPtrLeft = localData + haloSize * localBufferWidth + haloSize; // Pointer to [h][h]
    float* recvPtrLeft = localData + haloSize * localBufferWidth;
    // Pointer to [h][0] (start of left halo, skipping top halo row)

    MPI_Irecv(recvPtrLeft, 1, vertical_strip_type, mLeftRank, recv_tag, gridComm, &requests[4]);
    MPI_Isend(sendPtrLeft, 1, vertical_strip_type, mLeftRank, send_tag, gridComm, &requests[5]);

    // --- 4. Communication with RIGHT neighbor (mRightRank) ---
    // Send: Right boundary of the active region [h..h+H-1][h+W-h..h+W-1]
    // Recv: Into the right halo region [h..h+H-1][h+W..h+W+h-1]
    float* sendPtrRight =
        localData + haloSize * localBufferWidth + (haloSize + local_active_width - haloSize);
    // Pointer to [h][h+W-h] (start of right active edge)
    float* recvPtrRight = localData + haloSize * localBufferWidth + (haloSize + local_active_width);
    // Pointer to [h][h+W] (start of right halo, skipping top halo row)

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
    int localBufferWidth = local_active_width + 2 * haloSize; // Total width of the local buffer

    MPI_Aint lbw = (MPI_Aint)localBufferWidth;
    MPI_Aint h = (MPI_Aint)haloSize;
    MPI_Aint H = (MPI_Aint)local_active_height;
    MPI_Aint W = (MPI_Aint)local_active_width;

    // Use MPI_Put to send my active boundary to the neighbor's halo

    // --- Communication with LEFT neighbor (mLeftRank) ---
    // Put my Left active boundary [h..h+H-1][h..h+h-1] to neighbor's Right halo [h..h+H-1][h+W..h+W+h-1]
    if (mLeftRank != MPI_PROC_NULL) {
        float* origin_addr = localData + h * lbw + h; // Pointer to [h][h] (start of Left active boundary)
        MPI_Aint target_disp = h * lbw + (h + W); // Displacement to [h][h+W] (start of Right halo on neighbor)
        MPI_Put(origin_addr, 1, vertical_strip_type, mLeftRank, target_disp, 1, vertical_strip_type, window);
    }

    // --- Communication with RIGHT neighbor (mRightRank) ---
    // Put my Right active boundary [h..h+H-1][h+W-h..h+W-1] to neighbor's Left halo [h..h+H-1][0..h-1]
    if (mRightRank != MPI_PROC_NULL) {
        float* origin_addr = localData + h * lbw + (h + W - h);
        // Pointer to [h][h+W-h] (start of Right active boundary)
        MPI_Aint target_disp = h * lbw; // Displacement to [h][0] (start of Left halo on neighbor)
        MPI_Put(origin_addr, 1, vertical_strip_type, mRightRank, target_disp, 1, vertical_strip_type, window);
    }

    // --- Communication with TOP neighbor (mTopRank) ---
    // Put my Top active boundary [h..h+h-1][h..h+W-1] to neighbor's Bottom halo [h+H..h+H+h-1][h..h+W-1]
    if (mTopRank != MPI_PROC_NULL) {
        float* origin_addr = localData + h * lbw + h; // Pointer to [h][h] (start of Top active boundary)
        MPI_Aint target_disp = (h + H) * lbw + h; // Displacement to [h+H][h] (start of Bottom halo on neighbor)
        MPI_Put(origin_addr, 1, horizontal_strip_type, mTopRank, target_disp, 1, horizontal_strip_type, window);
    }

    // --- Communication with BOTTOM neighbor (mBottomRank) ---
    // Put my Bottom active boundary [h+H-h..h+H-1][h..h+W-1] to neighbor's Top halo [0..h-1][h..h+W-1]
    if (mBottomRank != MPI_PROC_NULL) {
        float* origin_addr = localData + (h + H - h) * lbw + h;
        // Pointer to [h+H-h][h] (start of Bottom active boundary)
        MPI_Aint target_disp = h;
        // Displacement to [0][h] (start of Top halo on neighbor, accounting for left halo size)
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
    // MPI_Win_fence performs collective synchronization for active target RMA
    MPI_Win_fence(0, window);
}

void ParallelHeatSolver::printLocalTilesWithoutHalo() {
    std::stringstream ss;

    ss << "\nRank: " << mWorldRank << " - Active tile area (mLocalTileTemperature[0]); " << mLocalTileSize[0] <<
        "x" << mLocalTileSize[1] << "\n";
    const int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;

    ss << std::fixed << std::setprecision(1);

    // Iterate only over the active area of the tile
    for (int y = 0; y < mLocalTileSize[1]; ++y) {
        for (int x = 0; x < mLocalTileSize[0]; ++x) {
            // Calculate index in the linear buffer, adjusting for halo offset
            size_t index = static_cast<size_t>(y + haloZoneSize) * localBufferWidth
                + static_cast<size_t>(x + haloZoneSize);
            ss << mLocalTileTemperature[0].at(index);
            ss << " ";
        }
        ss << "\n";
    }
    std::cout << ss.str();
}

void ParallelHeatSolver::printLocalTilesWithHalo() {
    std::stringstream ss;

    const int totalHeight = mLocalTileSize[1] + 2 * haloZoneSize;
    const int totalWidth = mLocalTileSize[0] + 2 * haloZoneSize;
    const int localBufferWidth = totalWidth;

    ss << "\nRank: " << mWorldRank << " - Full local buffer (mLocalTileTemperature[0]); "
        << totalWidth << "x" << totalHeight << " (incl. halo)\n";

    ss << std::fixed << std::setprecision(1);

    // Iterate over the entire buffer, including halo zones
    for (int y = 0; y < totalHeight; ++y) {
        for (int x = 0; x < totalWidth; ++x) {
            size_t index = static_cast<size_t>(y) * localBufferWidth + static_cast<size_t>(x);
            ss << mLocalTileTemperature[0].at(index);
            ss << " ";
        }
        ss << "\n";
    }
    std::cout << ss.str();
}

void ParallelHeatSolver::printGlobalGridAligned() const {
    // Only root process performs the print
    if (mWorldRank != ROOT) {
        return;
    }
    const auto* globalData = mMaterialProps.getInitialTemperature().data();

    if (globalData == nullptr) {
        std::fprintf(stderr, "[Rank %d] Error: printGlobalGridAligned called with nullptr!\n", mWorldRank);
        return;
    }

    const size_t edgeSize = mMaterialProps.getEdgeSize();
    if (edgeSize == 0) {
        std::printf("[Rank %d] Global grid size is 0.\n", mWorldRank);
        return;
    }

    std::stringstream ss;
    ss << "\n[Rank " << mWorldRank << "] Global temperature grid ("
        << edgeSize << "x" << edgeSize << ", aligned):\n";

    // Formatting settings
    const int precision = 1;
    const int fieldWidth = 8;

    ss << std::fixed << std::setprecision(precision);

    // Iterate through the global grid
    for (size_t y = 0; y < edgeSize; ++y) {
        for (size_t x = 0; x < edgeSize; ++x) {
            size_t index = y * edgeSize + x;
            ss << std::setw(fieldWidth);
            ss << globalData[index];
        }
        ss << "\n";
    }
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

    // Scatter initial temperature data into the first local buffer (newIdx for iter=-1)
    float* localTemperatureBuffer = mLocalTileTemperature[0].data();
    localDomainMap<float>(globalTemperatures, localTemperatureBuffer);

    // Scatter material properties (float)
    float* localMaterialPropBuffer = mLocalTileMaterialProp.data();
    localDomainMap<float>(globalDomainParameters, localMaterialPropBuffer);

    // Scatter material map (int)
    int* localMaterialMapBuffer = mLocalTileMaterial.data();
    localDomainMap<int>(globalDomainMap, localMaterialMapBuffer);


    /**********************************************************************************************************************/
    /* Exchange halo zones of initial domain temperature and parameters using P2P communication. Wait for them to finish. */
    /**********************************************************************************************************************/

    computeHaloZones(mLocalTileTemperature[0].data(), mLocalTileTemperature[0].data());
    // Compute into the same buffer initially? Or perhaps into buffer 1 using 0?
    // Let's assume computeHaloZones computes the *new* buffer from the *old* buffer.
    // For iter=0, oldIdx=0, newIdx=1. Initial scatter is into 0.
    // Need to copy 0 -> 1 first, then compute halos into 1 from 0.
    std::copy(mLocalTileTemperature[0].begin(), mLocalTileTemperature[0].end(), mLocalTileTemperature[1].begin());
    // Now compute halo zones for buffer 1 (new) from buffer 0 (old)
    computeHaloZones(mLocalTileTemperature[0].data(), mLocalTileTemperature[1].data());

    // Exchange initial temperatures (buffer 1) and properties (buffer 0 - they don't change)
    // Start halo exchange for initial temperature (buffer 1)
    startHaloExchangeP2P(mLocalTileTemperature[1].data(), requestsP2P);
    // Start halo exchange for material properties (buffer 0) - they need halo values too
    startHaloExchangeP2P(mLocalTileMaterialProp.data(), paramsRequests);

    // Wait for both exchanges to complete
    awaitHaloExchangeP2P(requestsP2P);
    awaitHaloExchangeP2P(paramsRequests);


    // Scatter initial data into mLocalTileTemperature[0]
    localDomainMap<float>(globalTemperatures, mLocalTileTemperature[0].data());
    // Scatter props/map (already done above)

    // Exchange initial halos for mLocalTileTemperature[0]
    startHaloExchangeP2P(mLocalTileTemperature[0].data(), requestsP2P);
    // Properties/Map halos only need to be exchanged once as they don't change
    startHaloExchangeP2P(mLocalTileMaterialProp.data(), paramsRequests);
    // This was already done above, is it needed again? No. It's part of init.

    // Need to wait for the initial temperature halo exchange before the first iteration
    awaitHaloExchangeP2P(requestsP2P);
    awaitHaloExchangeP2P(paramsRequests); // Wait for initial props/map halos too

    // The initial temperature data with valid halos is now in mLocalTileTemperature[0].
    // mLocalTileTemperature[1] is currently uninitialized or holds a copy *without* halos.
    // The loop will compute into mLocalTileTemperature[1] from mLocalTileTemperature[0].
    // No initial copy to buffer 1 needed here based on the loop structure.


    double startTime = MPI_Wtime();

    // Start main iterative simulation loop.
    for (std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter) {
        // oldIdx has temperatures with halos from the previous iteration
        const std::size_t oldIdx = iter % 2;
        // newIdx is where the new temperatures will be computed
        const std::size_t newIdx = (iter + 1) % 2;

        /**********************************************************************************************************************/
        /*                            Compute and exchange halo zones using P2P or RMA.                                       */
        /**********************************************************************************************************************/

        // Compute the boundary regions of the *active* area for the *new* buffer (newIdx).
        // These values are computed using the *old* buffer (oldIdx) which contains halos
        // from the previous iteration. These computed boundaries are the values that
        // will be sent to neighbors in this iteration's exchange.
        computeHaloZones(mLocalTileTemperature[oldIdx].data(), mLocalTileTemperature[newIdx].data());

        if (mSimulationProps.isRunParallelP2P()) {
            // Start P2P halo exchange: send boundaries of active area from newIdx,
            // receive into halo area of newIdx.
            startHaloExchangeP2P(mLocalTileTemperature[newIdx].data(), requestsP2P);
        }
        else if (mSimulationProps.isRunParallelRMA()) {
            // Start RMA halo exchange: put boundaries of active area from newIdx
            // into neighbor's halo area in their newIdx buffer.
            // Start an RMA epoch (Active Target Synchronization)
            MPI_Win_fence(0, m_rma_win[newIdx]);
            startHaloExchangeRMA(mLocalTileTemperature[newIdx].data(), m_rma_win[newIdx]);
        }

        /**********************************************************************************************************************/
        /*                           Compute the rest of the tile. Use updateTile method.                                     */
        /**********************************************************************************************************************/

        // Compute the inner active region (excluding boundaries computed in computeHaloZones)
        // for the *new* buffer (newIdx). This computation uses values from the *old* buffer (oldIdx).
        // This part of the computation can be done concurrently with the halo exchange.

        const int local_active_width = mLocalTileSize[0];
        const int local_active_height = mLocalTileSize[1];
        const int h_size = haloZoneSize;

        // Calculate dimensions and offsets for the INNER active region
        const std::size_t inner_offset_x = static_cast<std::size_t>(h_size +
            h_size); // Offset from left edge of buffer (skipping left halo and left boundary)
        const std::size_t inner_offset_y = static_cast<std::size_t>(h_size +
            h_size); // Offset from top edge of buffer (skipping top halo and top boundary)
        const std::size_t inner_size_x = static_cast<std::size_t>(local_active_width -
            2 * h_size); // Width of the inner region (excluding left/right boundaries)
        const std::size_t inner_size_y = static_cast<std::size_t>(local_active_height -
            2 * h_size); // Height of the inner region (excluding top/bottom boundaries)
        const std::size_t stride = static_cast<std::size_t>(local_active_width + 2 * h_size); // Total buffer width

        // Compute the inner region if it's non-empty
        if (inner_size_x > 0 && inner_size_y > 0) {
            updateTile(mLocalTileTemperature[oldIdx].data(),
                       mLocalTileTemperature[newIdx].data(),
                       mLocalTileMaterialProp.data(),
                       mLocalTileMaterial.data(),
                       inner_offset_x, // offsetX in buffer coordinates
                       inner_offset_y, // offsetY in buffer coordinates
                       inner_size_x, // sizeX
                       inner_size_y, // sizeY
                       stride); // stride
        }

        /**********************************************************************************************************************/
        /*                            Wait for all halo zone exchanges to finalize.                                           */
        /**********************************************************************************************************************/

        if (mSimulationProps.isRunParallelP2P()) {
            awaitHaloExchangeP2P(requestsP2P);
        }
        else if (mSimulationProps.isRunParallelRMA()) {
            // End the RMA epoch (Active Target Synchronization)
            awaitHaloExchangeRMA(m_rma_win[newIdx]);
        }

        // After waiting, the halo regions of mLocalTileTemperature[newIdx] are now filled with
        // the boundary values from the neighbors' active areas from this iteration.
        // This buffer (newIdx) is now ready to be the oldIdx for the next iteration.

        if (shouldStoreData(iter)) {
            /**********************************************************************************************************************/
            /*                          Store the data into the output file using parallel or sequential IO.                      */
            /**********************************************************************************************************************/

            if (mSimulationProps.useParallelIO()) {
                // Store the *active area* of the local tile (which is in newIdx)
                storeDataIntoFileParallel(mFileHandle, iter, mLocalTileTemperature[newIdx].data());
            }
            else {
                // Gather the *active area* of the local tile (from newIdx) into outResult on root
                gatherTiles<float>(mLocalTileTemperature[newIdx].data(), outResult.data());
                // Store the gathered global data (only on root)
                storeDataIntoFileSequential(mFileHandle, iter, outResult.data());
            }
        }

        if (shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature()) {
            /**********************************************************************************************************************/
            /*                 Compute and print middle column average temperature and print progress report.                     */
            /**********************************************************************************************************************/
            // Compute the average temperature in the middle column using the newIdx buffer
            float middleColAvgTemp = computeMiddleColumnAverageTemperatureParallel(
                mLocalTileTemperature[newIdx].data());

            // Only rank 0 within the middle column communicator prints the progress report
            int rank;
            MPI_Comm_rank(avgTempComm, &rank);

            if (rank == 0)
                printProgressReport(iter, middleColAvgTemp);
        }
    }

    // The result of the simulation is in the buffer indicated by the parity of the last iteration number.
    const std::size_t resIdx = mSimulationProps.getNumIterations() % 2;

    double elapsedTime = MPI_Wtime() - startTime;

    /**********************************************************************************************************************/
    /*                                     Gather final domain temperature.                                               */
    /**********************************************************************************************************************/

    // Gather the active area of the final local tile (from resIdx buffer)
    gatherTiles<float>(mLocalTileTemperature[resIdx].data(), outResult.data());

    /**********************************************************************************************************************/
    /*           Compute (sequentially) and report final middle column temperature average and print final report.        */
    /**********************************************************************************************************************/

    // Only root computes and prints the final sequential report
    if (mWorldRank == 0) {
        auto avg = computeMiddleColumnAverageTemperatureSequential(outResult.data());
        printFinalReport(elapsedTime, avg);
    }
}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const {
    /**********************************************************************************************************************/
    /*                Return true if rank should compute middle column average temperature.                               */
    /**********************************************************************************************************************/
    // A rank should compute the middle column average if it is part of the avgTempComm
    return (avgTempComm != MPI_COMM_NULL);
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureParallel(const float* localData) const {
    /**********************************************************************************************************************/
    /*                  Implement parallel middle column average temperature computation.                                 */
    /*                      Use OpenMP directives to accelerate the local computations.                                   */
    /**********************************************************************************************************************/

    float local_sum = 0.0;

    int nRanksInComm = 0; // Default value if avgTempComm is NULL
    if (avgTempComm != MPI_COMM_NULL) {
        MPI_Comm_size(avgTempComm, &nRanksInComm);
    }
    else {
        // If this rank is not in the middle column, its local sum is 0 and it shouldn't participate in reduction.
        // The calling code checks shouldComputeMiddleColumnAverageTemperature() before calling this.
        // However, to be safe, return 0 if not in the comm.
        return 0.0f;
    }

    const int localBufferWidth = mLocalTileSize[0] + 2 * haloZoneSize;
    // The middle column within the local tile's active area
    const int middleColIndexLocal = 0; // always odd grid size, so middle column is 0

    // Use OpenMP to parallelize local summation over the local tile's middle column
#pragma omp parallel for reduction(+:local_sum) schedule(static)
    for (std::size_t y = haloZoneSize; y < mLocalTileSize[1] + haloZoneSize; ++y) {
        // Index: row_in_buffer * buffer_width + column_in_buffer
        local_sum += localData[y * localBufferWidth + haloZoneSize + middleColIndexLocal];
    }

    // If there's only one rank in the middle column communicator (e.g., total Y decomposition is 1)
    if (nRanksInComm == 1) {
        // Total sum is just the local sum, divide by the number of points (local tile height)
        return local_sum / static_cast<float>(mLocalTileSize[1]);
    }

    float global_sum = 0.0;
#ifdef USE_KAMPING_LIB
    // --- KaMPing ---

    m_kamping_avg_comm.reduce(
        kamping::send_buf(local_sum),
        kamping::recv_buf(global_sum),
        kamping::op([](auto a, auto b) { return a + b; },
        kamping::ops::non_commutative)
        );
#else
    // Perform a reduction (sum) across all ranks in the middle column communicator
    MPI_Reduce(
        &local_sum, // Send buffer: Local sum
        &global_sum, // Receive buffer: Global sum (only relevant on root of avgTempComm)
        1, // Number of elements to reduce
        MPI_FLOAT, // Datatype of elements
        MPI_SUM, // Reduction operation
        0, // Root rank within the avgTempComm (rank 0 in this specific communicator)
        avgTempComm // Communicator for middle column ranks
    );
#endif /* USE_KAMPING_LIB */

    // Calculate the global average temperature (only root of avgTempComm will have the correct global_sum)
    // The total number of points in the middle column is (global grid height) * 1.
    // In a decomposed grid, this is sum of local tile heights across the middle column ranks.
    // Since all local tiles have height mLocalTileSize[1] and there are nRanksInComm processes,
    // total points = mLocalTileSize[1] * nRanksInComm.
    return global_sum / static_cast<float>(mLocalTileSize[1] * nRanksInComm);
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureSequential(const float* globalData) const {
    /**********************************************************************************************************************/
    /*                  Implement sequential middle column average temperature computation.                               */
    /*                      Use OpenMP directives to accelerate the local computations.                                   */
    /**********************************************************************************************************************/


    double middleColSum = 0.0; // Use double for summation to maintain precision
    const std::size_t edgeSize = mMaterialProps.getEdgeSize();

    // Check for empty grid to prevent division by zero
    if (edgeSize == 0) {
        return 0.0f;
    }

    const std::size_t middleColIndex = edgeSize / 2; // Index of the middle column in the global grid

    // Parallelize the loop over the rows using OpenMP
#pragma omp parallel for reduction(+:middleColSum) schedule(static)
    for (std::size_t y = 0; y < edgeSize; ++y) {
        // Calculate the linear index for the point in the middle column for row y
        std::size_t index = y * edgeSize + middleColIndex;
        middleColSum += globalData[index];
    }

    // Calculate the average temperature by dividing the total sum by the number of points in the column (edgeSize)
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
    // This function is only called on root (handled by shouldStoreData and run logic).
    storeDataIntoFile(fileHandle, iteration, globalData);
}

void ParallelHeatSolver::openOutputFileParallel() {
#ifdef H5_HAVE_PARALLEL
    Hdf5PropertyListHandle faplHandle{};

    /**********************************************************************************************************************/
    /*                          Open output HDF5 file for parallel access with alignment.                                 */
    /*      Set up faplHandle to use MPI-IO and alignment. The handle will automatically release the resource.            */
    /**********************************************************************************************************************/

    // Create a file access property list
    faplHandle = H5Pcreate(H5P_FILE_ACCESS);
    // Set the file access property list to use MPI-IO
    H5Pset_fapl_mpio(faplHandle, gridComm, MPI_INFO_NULL);

    // Set alignment for file access (optional, but can improve performance)
    hsize_t alignment = 1024 * 1024; // 1MB alignment
    H5Pset_alignment(faplHandle, 0, alignment);

    // Create the HDF5 file collectively
    mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                            H5F_ACC_TRUNC, // Truncate file if it exists
                            H5P_DEFAULT, // Link creation property list
                            faplHandle); // File access property list

    if (!mFileHandle.valid()) {
        throw std::ios::failure("Cannot create output file!");
    }
#else
    throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}

void ParallelHeatSolver::storeDataIntoFileParallel(hid_t fileHandle,
                                                   std::size_t iteration,
                                                   const float* localData) {
    if (fileHandle == H5I_INVALID_HID) {
        return;
    }

#ifdef H5_HAVE_PARALLEL
    // Global grid dimensions for the HDF5 dataspace
    std::array gridSize{
        static_cast<hsize_t>(mMaterialProps.getEdgeSize()), // Y dimension (rows)
        static_cast<hsize_t>(mMaterialProps.getEdgeSize()) // X dimension (columns)
    };

    // Create new HDF5 group for this timestep
    std::string groupName = "Timestep_" + std::to_string(iteration / mSimulationProps.getWriteIntensity());
    Hdf5GroupHandle groupHandle(H5Gcreate(fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    {
        /**********************************************************************************************************************/
        /*                                Compute the tile offsets and sizes.                                                 */
        /*               Note that the X and Y coordinates are swapped (but data not altered).                                */
        /**********************************************************************************************************************/
        // Size of the active tile area for this process
        std::array<hsize_t, 2> tileSize{
            static_cast<hsize_t>(mLocalTileSize[1]), // Y dimension (rows)
            static_cast<hsize_t>(mLocalTileSize[0]) // X dimension (columns)
        };
        // Offset to the active area within the local buffer (memory)
        std::array<hsize_t, 2> localDataTileOffset{haloZoneSize, haloZoneSize}; // {offsetY, offsetX}

        // Calculate the offset of this process's tile in the GLOBAL grid (file dataspace).
        // Note: HDF5 dataspace is {rows, columns} i.e., {Y, X}.
        // myCoorsGrid is {X-coord, Y-coord}.
        // The X-coordinate in myCoorsGrid corresponds to the column offset.
        // The Y-coordinate in myCoorsGrid corresponds to the row offset.
        hsize_t offsetInGlobalX = static_cast<hsize_t>(myCoorsGrid[0]) * mLocalTileSize[0];
        hsize_t offsetInGlobalY = static_cast<hsize_t>(myCoorsGrid[1]) * mLocalTileSize[1];

        // The comment says "X and Y coordinates are swapped".
        // This implies myCoorsGrid[0] (X) is used for the Y (row) offset, and myCoorsGrid[1] (Y) for the X (column) offset.
        // Adhering to the comment:
        hsize_t offsetX_swapped = static_cast<hsize_t>(myCoorsGrid[1]) * mLocalTileSize[0];
        hsize_t offsetY_swapped = static_cast<hsize_t>(myCoorsGrid[0]) * mLocalTileSize[1];

        // Offset of the tile within the global dataset (file)

        hsize_t offsetInGlobalY_standard = static_cast<hsize_t>(myCoorsGrid[1]) * mLocalTileSize[1];
        // Y-coord * tile_height
        hsize_t offsetInGlobalX_standard = static_cast<hsize_t>(myCoorsGrid[0]) * mLocalTileSize[0];
        // X-coord * tile_width

        // Use the standard offset
        std::array<hsize_t, 2> tileOffsetInGlobal{offsetInGlobalY_standard, offsetInGlobalX_standard};
        // std::array<hsize_t, 2> tileOffsetInGlobal{offsetY_swapped, offsetX_swapped}; // {offsetY, offsetX} - based on comment's swap


        // Create new dataspace for the dataset in the file (represents the whole global grid).
        static constexpr std::string_view dataSetName{"Temperature"};
        Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));

        Hdf5PropertyListHandle datasetPropListHandle{};
        /**********************************************************************************************************************/
        /*                            Create dataset property list to set up chunking.                                        */
        /*                Set up chunking for collective write operation in datasetPropListHandle variable.                   */
        /**********************************************************************************************************************/
        // Create a dataset creation property list
        datasetPropListHandle = H5Pcreate(H5P_DATASET_CREATE);
        // Set chunking dimensions to the size of the local tile
        H5Pset_chunk(datasetPropListHandle, 2, tileSize.data());

        // Create the dataset
        Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                                  H5T_NATIVE_FLOAT, // Data type
                                                  dataSpaceHandle, // Dataspace describing the dataset
                                                  H5P_DEFAULT, // Link creation property list
                                                  datasetPropListHandle,
                                                  // Dataset creation property list (for chunking)
                                                  H5P_DEFAULT)); // Dataset access property list

        Hdf5DataspaceHandle memSpaceHandle{};
        /**********************************************************************************************************************/
        /*                Create memory dataspace representing tile in the memory (set up memSpaceHandle).                    */
        /**********************************************************************************************************************/
        // Create a memory dataspace representing the local buffer including halo zones
        std::array<hsize_t, 2> tileSizeWithHaloZones{
            static_cast<hsize_t>(mLocalTileSize[1] + 2 * haloZoneSize), // Y dimension (rows)
            static_cast<hsize_t>(mLocalTileSize[0] + 2 * haloZoneSize) // X dimension (columns)
        };
        memSpaceHandle = H5Screate_simple(2, tileSizeWithHaloZones.data(), nullptr);

        /**********************************************************************************************************************/
        /*              Select inner part of the tile in memory and matching part of the dataset in the file                  */
        /*                           (given by position of the tile in global domain).                                        */
        /**********************************************************************************************************************/
        // Select the active area within the memory buffer
        H5Sselect_hyperslab(memSpaceHandle, H5S_SELECT_SET, localDataTileOffset.data(), nullptr, tileSize.data(),
                            nullptr);
        // Select the corresponding tile area within the file dataset (global dataspace)
        H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, tileOffsetInGlobal.data(), nullptr, tileSize.data(),
                            nullptr);

        Hdf5PropertyListHandle propListHandle{};

        /**********************************************************************************************************************/
        /*              Perform collective write operation, writting tiles from all processes at once.                        */
        /*                                   Set up the propListHandle variable.                                              */
        /**********************************************************************************************************************/
        // Create a data transfer property list
        propListHandle = H5Pcreate(H5P_DATASET_XFER);
        // Set the data transfer mode to collective
        H5Pset_dxpl_mpio(propListHandle, H5FD_MPIO_COLLECTIVE);

        // Perform the collective write
        H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, dataSpaceHandle, propListHandle, localData);
    }

    {
        // Store attribute with current iteration number in the group.
        static constexpr std::string_view attributeName{"Time"};
        Hdf5DataspaceHandle dataSpaceHandle(H5Screate(H5S_SCALAR)); // Scalar dataspace for attribute
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
