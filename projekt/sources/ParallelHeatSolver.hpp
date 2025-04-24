/**
 * @file    ParallelHeatSolver.hpp
 *
 * @author  Name Surname <xlogin00@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-02-23
 */

#ifndef PARALLEL_HEAT_SOLVER_HPP
#define PARALLEL_HEAT_SOLVER_HPP

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include <mpi.h>

#include "AlignedAllocator.hpp"
#include "Hdf5Handle.hpp"
#include "HeatSolverBase.hpp"

#ifdef USE_KAMPING_LIB
    #include <kamping/communicator.hpp>
    #include <kamping/collectives/reduce.hpp>
    #include <kamping/communicator.hpp>
    #include <kamping/data_buffer.hpp>
    #include <kamping/environment.hpp>
    #include <kamping/mpi_ops.hpp>
    #include <kamping/named_parameters.hpp>
#endif


/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 2D block grid decomposition.
 */
class ParallelHeatSolver : public HeatSolverBase {
public:
    /**
     * @brief Constructor - Initializes the solver. This includes:
     *        - Construct 2D grid of tiles.
     *        - Create MPI datatypes used in the simulation.
     *        - Open SEQUENTIAL or PARALLEL HDF5 file.
     *        - Allocate data for local tiles.
     *
     * @param simulationProps Parameters of simulation - passed into base class.
     * @param materialProps   Parameters of material - passed into base class.
     */
    ParallelHeatSolver(const SimulationProperties& simulationProps, const MaterialProperties& materialProps);

    /// @brief Inherit constructors from the base class.
    using HeatSolverBase::HeatSolverBase;

    /**
     * @brief Destructor - Releases all resources allocated by the solver.
     */
    virtual ~ParallelHeatSolver() override;

    /// @brief Inherit assignment operator from the base class.
    using HeatSolverBase::operator=;

    /**
     * @brief Run main simulation loop.
     * @param outResult Output array which is to be filled with computed temperature values.
     *                  The vector is pre-allocated and its size is given by dimensions
     *                  of the input file (edgeSize*edgeSize).
     *                  NOTE: The vector is allocated (and should be used) *ONLY*
     *                        by master process (rank 0 in MPI_COMM_WORLD)!
     */
    virtual void run(std::vector<float, AlignedAllocator<float>>& outResult) override;

protected:
private:
    /**
     * @brief Get type of the code.
     * @return Returns type of the code.
     */
    std::string_view getCodeType() const override;

    /**
     * @brief Initialize the grid topology.
     */
    void initGridTopology();

    /**
     * @brief Deinitialize the grid topology.
     */
    void deinitGridTopology();

    /**
     * @brief Initialize variables and MPI datatypes for data scattering and gathering.
     */
    void initDataDistribution();

    /**
     * @brief Deinitialize variables and MPI datatypes for data scattering and gathering.
     */
    void deinitDataDistribution();

    /**
     * @brief Allocate memory for local tiles.
     */
    void allocLocalTiles();

    /**
     * @brief Deallocate memory for local tiles.
     */
    void deallocLocalTiles();

    /**
     * @brief Initialize variables and MPI datatypes for halo exchange.
     */
    void initHaloExchange();

    /**
     * @brief Deinitialize variables and MPI datatypes for halo exchange.
     */
    void deinitHaloExchange();

    /**
     * @brief Scatter global data to local tiles.
     * @tparam T Type of the data to be scattered. Must be either float or int.
     * @param globalData Global data to be scattered.
     * @param localData  Local data to be filled with scattered values.
     */
    template <typename T>
    void localDomainMap(const T* globalData, T* localData);

    /**
     * @brief Gather local tiles to global data.
     * @tparam T Type of the data to be gathered. Must be either float or int.
     * @param localData  Local data to be gathered.
     * @param globalData Global data to be filled with gathered values.
     */
    template <typename T>
    void gatherTiles(const T* localData, T* globalData);

    /**
     * @brief Compute temperature of the next iteration in the halo zones.
     * @param oldTemp Old temperature values.
     * @param newTemp New temperature values.
     */
    void computeHaloZones(const float* oldTemp, float* newTemp);

    /**
     * @brief Start non-blocking halo exchange using point-to-point communication.
     * @param localData Local data buffer containing active area and halo zones.
     * @param request   Array of MPI_Request objects to be filled with requests for pending operations.
     */
    void startHaloExchangeP2P(float* localData, std::array<MPI_Request, 8>& request);

    /**
     * @brief Wait for all pending non-blocking halo exchanges started with P2P communication to finalize.
     * @param request Array of MPI_Request objects to be awaited.
     */
    void awaitHaloExchangeP2P(std::array<MPI_Request, 8>& request);

    /**
     * @brief Start non-blocking halo exchange using RMA communication.
     * @param localData Local data buffer containing active area and halo zones.
     * @param window    MPI_Win object associated with the local buffer for RMA communication.
     */
    void startHaloExchangeRMA(float* localData, MPI_Win window);

    /**
     * @brief Wait for all pending non-blocking halo exchanges started with RMA communication to finalize.
     * @param window MPI_Win object associated with the local buffer.
     */
    void awaitHaloExchangeRMA(MPI_Win window);

    /**
     * @brief Prints the active area of the local temperature tile (without halo zones).
     *        Used for debugging.
     */
    void printLocalTilesWithoutHalo();

    /**
     * @brief Prints the entire local temperature tile buffer (including halo zones).
     *        Used for debugging.
     */
    void printLocalTilesWithHalo();

    /**
     * @brief Prints the global initial temperature grid in an aligned format (only on root).
     *        Used for debugging.
     */
    void printGlobalGridAligned() const;

    /**
     * @brief Computes global average temperature of middle column across
     *        processes in "mGridMiddleColComm" communicator.
     *        NOTE: All ranks in the communicator *HAVE* to call this method.
     *              Uses OpenMP for local sum.
     * @param localData Data of the local tile buffer (including halo zones).
     * @return Returns average temperature over middle of all tiles in the communicator.
     */
    float computeMiddleColumnAverageTemperatureParallel(const float* localData) const;

    /**
     * @brief Computes global average temperature of middle column of the domain
     *        using values collected to MASTER rank.
     *        NOTE: Only single MASTER (rank 0) should call this method.
     *              Uses OpenMP for sequential sum.
     * @param globalData Simulation state collected to the MASTER rank.
     * @return Returns the average temperature.
     */
    float computeMiddleColumnAverageTemperatureSequential(const float* globalData) const;

    /**
     * @brief Opens output HDF5 file for sequential access by MASTER rank only.
     *        NOTE: Only MASTER (rank = 0) should call this method.
     */
    void openOutputFileSequential();

    /**
     * @brief Stores current state of the simulation into the output file sequentially.
     *        NOTE: Only MASTER (rank = 0) should call this method.
     * @param fileHandle HDF5 file handle to be used for the writting operation.
     * @param iteration  Integer denoting current iteration number.
     * @param globalData Square 2D array of edgeSize x edgeSize elements containing
     *                   simulation state to be stored in the file.
     */
    void storeDataIntoFileSequential(hid_t fileHandle, std::size_t iteration, const float* globalData);

    /**
     * @brief Opens output HDF5 file for parallel/cooperative access.
     *        NOTE: This method *HAS* to be called from all processes in the communicator.
     */
    void openOutputFileParallel();

    /**
     * @brief Stores current state of the simulation into the output file collectively using parallel HDF5.
     *        NOTE: All processes which opened the file HAVE to call this method collectively.
     * @param fileHandle HDF5 file handle to be used for the writting operation.
     * @param iteration  Integer denoting current iteration number.
     * @param localData  Local 2D array (tile) buffer including halo zones. The method stores
     *                   only the active area data.
     */
    void storeDataIntoFileParallel(hid_t fileHandle, std::size_t iteration, const float* localData);

    /**
     * @brief Determines if the process is part of the middle column communicator and should compute average temperature.
     * @return Returns true if the process should compute middle column average temperature.
     */
    bool shouldComputeMiddleColumnAverageTemperature() const;

    /// @brief Code type string identifier.
    static constexpr std::string_view codeType{"par"};

    /// @brief Size of the halo zone (number of rows/columns added to each side of the active area).
    static constexpr std::size_t haloZoneSize{2};

    /// @brief Process rank in the global communicator (MPI_COMM_WORLD).
    int mWorldRank{};

    /// @brief Total number of processes in MPI_COMM_WORLD.
    int mWorldSize{};

    /// @brief Handle for the opened output HDF5 file (can be for sequential or parallel access).
    Hdf5FileHandle mFileHandle{};

    MPI_Comm gridComm = MPI_COMM_NULL; ///< MPI communicator for the 2D Cartesian grid topology.
    MPI_Comm avgTempComm = MPI_COMM_NULL;
    ///< MPI communicator specifically for ranks in the middle column of the grid, used for average temperature computation.

    /// @brief Edge size of the global computational grid.
    int mGridSizeEdge{};
    /// @brief Size of the active area of the local tile {width, height}.
    int mLocalTileSize[2]{};
    /// @brief MPI Datatype describing the layout of a tile within the global grid for float data.
    MPI_Datatype tileTypeFloat = MPI_DATATYPE_NULL;
    /// @brief MPI Datatype describing the layout of a tile within the global grid for int data.
    MPI_Datatype tileTypeInt = MPI_DATATYPE_NULL;
    /// @brief MPI Datatype describing the layout of the active area within the local buffer for int data.
    MPI_Datatype m_localActiveAreaTypeInt = MPI_DATATYPE_NULL;
    /// @brief MPI Datatype describing the layout of the active area within the local buffer for float data.
    MPI_Datatype m_localActiveAreaTypeFloat = MPI_DATATYPE_NULL;

    /// @brief MPI Datatype describing a horizontal strip (row(s)) within the local buffer, used for halo exchange.
    MPI_Datatype horizontal_strip_type = MPI_DATATYPE_NULL;
    /// @brief MPI Datatype describing a vertical strip (column(s)) within the local buffer, used for halo exchange.
    MPI_Datatype vertical_strip_type = MPI_DATATYPE_NULL;

    /// @brief Local buffer for material map data (int). Includes halo zones.
    std::vector<int, AlignedAllocator<int>> mLocalTileMaterial{};
    /// @brief Local buffer for material properties data (float). Includes halo zones.
    std::vector<float, AlignedAllocator<float>> mLocalTileMaterialProp{};
    /// @brief Two local buffers for temperature data (float). Used for double buffering. Each includes halo zones.
    std::array<std::vector<float, AlignedAllocator<float>>, 2> mLocalTileTemperature{};

    /// @brief Total number of processes in the grid communicator.
    int m_size;

    /// @brief Coordinates of this process in the 2D Cartesian grid {X-coord, Y-coord}.
    int myCoorsGrid[2];
    /// @brief Ranks of neighboring processes in the grid communicator {Top, Bottom, Right, Left}. MPI_PROC_NULL if no neighbor.
    int mTopRank, mBottomRank, mRightRank, mLeftRank;

    /// @brief MPI Windows for RMA communication, one for each temperature buffer.
    std::array<MPI_Win, 2> m_rma_win = {MPI_WIN_NULL, MPI_WIN_NULL};

    /**
     * @brief Helper function to get the rank of the current process within a given communicator.
     * @param c The MPI communicator.
     * @return The rank of the process in the communicator.
     */
    int rank(MPI_Comm& c) {
        int rank;
        MPI_Comm_rank(c, &rank);
        return rank;
    }

#ifdef USE_KAMPING_LIB
    kamping::Communicator<> m_kamping_avg_comm; ///< Kamping communicator wrapper for avgTempComm.
#endif

    /// @brief Rank of the root process (conventionally 0).
    static constexpr int ROOT = 0;
};

#endif /* PARALLEL_HEAT_SOLVER_HPP */
