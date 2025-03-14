/**
 * @file      scatter.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 * 
 * @author    David Bayer \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            ibayer@fit.vutbr.cz
 *
 * @brief     PC lab 2 / MPI Scatter
 *
 * @version   2023
 *
 * @date      23 February  2020, 15:45 (created) \n
 * @date      28 February  2021, 19:06 (created) \n
 * @date      14 February  2024, 11:58 (updated) \n
 *
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <mpi.h>

/// constant used for mpi printf
constexpr int MPI_ALL_RANKS = -1;
/// constant for root rank
constexpr int MPI_ROOT_RANK = 0;

//--------------------------------------------------------------------------------------------------------------------//
//                                            Helper function prototypes                                              //
//--------------------------------------------------------------------------------------------------------------------//
/// Wrapper to the C printf routine and specifies who can print (a given rank or all).
template<typename... Args>
void mpiPrintf(int who, const std::string_view format, Args... args);

/// Flush standard output.
void mpiFlush();

/// Parse command line parameters.
int parseParameters(int argc, char **argv);

/// Return MPI rank in a given communicator.
int mpiGetCommRank(const MPI_Comm &comm);

/// Return size of the MPI communicator.
int mpiGetCommSize(const MPI_Comm &comm);


//--------------------------------------------------------------------------------------------------------------------//
//                                                 Helper functions                                                   //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * C printf routine with selection which rank prints.
 * @param who    - which rank should print. If -1 then all prints.
 * @param format - format string.
 * @param ...    - other parameters.
 */
template<typename... Args>
void mpiPrintf(int who, const std::string_view format, Args... args) {
    if ((who == MPI_ALL_RANKS) || (who == mpiGetCommRank(MPI_COMM_WORLD))) {
        if constexpr (sizeof...(args) == 0) {
            std::printf("%s", std::data(format));
        } else {
            std::printf(std::data(format), args...);
        }
    }
}// end of mpiPrintf
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush stdout and call barrier to prevent message mixture.
 */
void mpiFlush() {
    std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // A known hack to correctly order writes
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}// end of mpiFlush
//----------------------------------------------------------------------------------------------------------------------

/**
 * Parse commandline - expecting a single parameter - test Id.
 * @param argc.
 * @param argv.
 * @return test id.
 */
int parseParameters(int argc, char **argv) {
    if (argc != 2) {
        mpiPrintf(MPI_ROOT_RANK, "!!!                   Please specify test number!                !!!\n");
        mpiPrintf(MPI_ROOT_RANK, "--------------------------------------------------------------------\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    return std::stoi(argv[1]);
}// end of parseParameters
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get MPI rank within the communicator.
 * @param [in] comm - actual communicator.
 * @return rank within the comm.
 */
int mpiGetCommRank(const MPI_Comm &comm) {
    int rank{};

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    return rank;
}// end of mpiGetCommRank
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get size of the communicator.
 * @param [in] comm - actual communicator.
 * @return number of ranks within the comm.
 */
int mpiGetCommSize(const MPI_Comm &comm) {
    int size{};

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    return size;
}// end of mpiGetCommSize
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//                                                   Main routine                                                     //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Main function
 */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // Set stdout to print out
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiPrintf(MPI_ROOT_RANK, "                             PPP Lab 2                               \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // Parse parameters
    const int testId = parseParameters(argc, argv);

    // Select test
    switch (testId) {
//--------------------------------------------------------------------------------------------------------------------//
//                          Example 1 - Dot product of the vector using Scatter + Reduction.                          //
//--------------------------------------------------------------------------------------------------------------------//
        case 1:  // Example 1 - Broadcast your name.
        {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 1 - Dot product of the vector using Scatter + Reduction     \n" \
                               "             balanced distribution \n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            constexpr int size = 128;

            // Global array and global result.
            std::vector<int> a{};
            std::vector<int> b{};

            int parResult{};
            int seqResult{};

            // Initialize arrays and calculate a sequential version.
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                a.resize(size);
                b.resize(size);

                std::random_device rd{};
                std::mt19937 gen{rd()};

                std::uniform_int_distribution<int> disA{0, 100};
                std::uniform_int_distribution<int> disB{0, 20};

                // Init array
                std::generate(std::begin(a), std::end(a), [&disA, &gen]() { return disA(gen); });
                std::generate(std::begin(b), std::end(b), [&disB, &gen]() { return disB(gen); });

                seqResult = std::transform_reduce(std::begin(a), std::end(a), std::begin(b), 0);
            }


            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                            Enter your code here                                              //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // 1. Define local partial result.

            int localResult{};



            // 2. Calculate local size of the arrays, each rank knows the global size.

            int localSize = size / mpiGetCommSize(MPI_COMM_WORLD);
            int globalsize = size;
            MPI_Bcast(&globalsize, 1, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);


            // 3. Allocate local arrays at every rank using std::vector.

            std::vector<int> localA(localSize);
            std::vector<int> localB(localSize);



            // 4. Scatter vector a and b into local vectors.

            MPI_Scatter(a.data(), localSize, MPI_INT, localA.data(), localSize, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);
            MPI_Scatter(b.data(), localSize, MPI_INT, localB.data(), localSize, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

            // 5. Calculate local dot product

            for (int i = 0; i < localSize; i++) {
                localResult += localA[i] * localB[i];
            }


            // 6. Reduce partial results back to the root rank.

            MPI_Reduce(&localResult, &parResult, 1, MPI_INT, MPI_SUM, MPI_ROOT_RANK, MPI_COMM_WORLD);



            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Print results.
            mpiPrintf(MPI_ROOT_RANK, " Seq result = %d\n", seqResult);
            mpiPrintf(MPI_ROOT_RANK, " Par result = %d\n", parResult);
            mpiPrintf(MPI_ROOT_RANK, " Status = %s\n", ((seqResult - parResult) == 0) ? "Ok" : "Fail");

            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        } //case 1

//--------------------------------------------------------------------------------------------------------------------//
//              Example 2 - Dot product of the vector using Scatter + Reduction - Unbalanced distribution             //
//--------------------------------------------------------------------------------------------------------------------//
        case 2: // Example 2 - Broadcast your name using 2 Sends and Recvs.
        {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 2 - Dot product of the vector using Scatter + Reduction     \n" \
                               "             unbalanced distribution \n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            constexpr int size = 127;

            // Global array and global result
            std::vector<int> a(size);
            std::vector<int> b(size);

            int parResult{};
            int seqResult{};

            // Initialize arrays and calculate a sequential version.
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                a.resize(size);
                b.resize(size);

                std::random_device rd{};
                std::mt19937 gen{rd()};

                std::uniform_int_distribution<int> disA{0, 100};
                std::uniform_int_distribution<int> disB{0, 20};

                // Init array
                std::generate(std::begin(a), std::end(a), [&disA, &gen]() { return disA(gen); });
                std::generate(std::begin(b), std::end(b), [&disB, &gen]() { return disB(gen); });

                seqResult = std::transform_reduce(std::begin(a), std::end(a), std::begin(b), 0);
            }


            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                        Enter your code here                                                  //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // 1. Define local partial result.


            // 2. Calculate chunkSize for each rank as (size / process count)

            // 3. Calculate reminder - how many elements remains


            // 4. Allocate arrays for sendCounts and displacements



            // 5.  Initialize sendCounts
            // <0        ... reminder)   = chunkSize + 1;
            // <reminder ... proc count) = chunkSize;




            // 6. Initialize displacements
            // displacements[0] = 0;
            // displacements[i] += displacements[i - 1]







            // 7. Allocate local array at every rank.



            // 8. Scatter vector a and b into local vectors using Scatterv




            // 9. Calculate local dot product



            // 10. Reduce partial results back to the root rank.



            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Print results.
            mpiPrintf(MPI_ROOT_RANK, " Seq result = %d\n", seqResult);
            mpiPrintf(MPI_ROOT_RANK, " Par result = %d\n", parResult);
            mpiPrintf(MPI_ROOT_RANK, " Status = %s\n", ((seqResult - parResult) == 0) ? "Ok" : "Fail");


            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        }


        default: {
            mpiPrintf(0, " !!!                     Unknown test number                     !!!\n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            break;
        }
    }// switch

    MPI_Finalize();
}// end of main
//----------------------------------------------------------------------------------------------------------------------

