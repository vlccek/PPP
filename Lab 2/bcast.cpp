/**
 * @file      bcast.cpp
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
 * @brief     PC lab 2 / MPI Broadcast
 *
 * @version   2023
 *
 * @date      23 February  2020, 11:13 (created) \n
 * @date      28 February  2023, 17:49 (revised) \n
 * @date      14 February  2024, 11:58 (updated) \n
 *
 */

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include <mpi.h>

/// constant used for mpi printf
constexpr int MPI_ALL_RANKS = -1;
/// constant for root rank
constexpr int MPI_ROOT_RANK = 0;

/// student's login
constexpr std::string_view studentLogin = "xvlkja07";

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
int mpiGetCommRank(MPI_Comm comm);

/// Return size of the MPI communicator.
int mpiGetCommSize(MPI_Comm comm);


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
int mpiGetCommRank(MPI_Comm comm) {
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
int mpiGetCommSize(MPI_Comm comm) {
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
//                              Example 1 - Broadcast your name to all ranks using Bcast                              //
//--------------------------------------------------------------------------------------------------------------------//
        case 1: {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 1 - Broadcast your name to all ranks using Bcast            \n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            // Define string and its length
            std::string myName{};
            int myNameLength{};

            // Enter your name at rank 0:
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                myName = studentLogin;
                myNameLength = myName.size();
            }

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                             Enter your code here                                             //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // 1. Broadcast the size of the string.
            MPI_Bcast(&myNameLength, 1, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

            // 2. Reserve buffer on the other ranks.
            myName.resize(myNameLength);

            // 3. Broadcast your name (use address of the first char in the string as a buffer address).
            MPI_Bcast(myName.data(), myNameLength, MPI_CHAR, MPI_ROOT_RANK, MPI_COMM_WORLD);

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Print your name
            mpiPrintf(MPI_ALL_RANKS, "Rank %d: %s (%d characters).\n",
                      mpiGetCommRank(MPI_COMM_WORLD), myName.c_str(), std::size(myName));

            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        } //case 1

//--------------------------------------------------------------------------------------------------------------------//
//                               Example 2 - Broadcast your name using 2 Sends and Recvs.                             //
//--------------------------------------------------------------------------------------------------------------------//
        case 2: {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 2 - Broadcast your name to all ranks using 2 Sends and Recvs\n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            // Define string and its length
            std::string myName{};
            int myNameLength{};

            // Enter your name at rank 0:
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                myName = studentLogin;
                myNameLength = myName.size();
            }


            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                             Enter your code here                                             //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // 1. Define tags for string length and string data.
            int tag_1 = 1;
            int tag_2 = 2;


            // 2. Root rank sends the size of the string and the string to all partners excluding itself.

            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                for (int i = 1; i < mpiGetCommSize(MPI_COMM_WORLD); i++) {
                    MPI_Send(&myNameLength, 1, MPI_INT, i, tag_1, MPI_COMM_WORLD);
                    MPI_Send(myName.c_str(), myNameLength, MPI_CHAR, i, tag_2, MPI_COMM_WORLD);
                }
            }


            //3. Other processes receive the length of the string, reserve space in myName and receive string.
            if (mpiGetCommRank(MPI_COMM_WORLD) != MPI_ROOT_RANK) {
                MPI_Recv(&myNameLength, 1, MPI_INT, MPI_ROOT_RANK, tag_1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                myName.resize(myNameLength);
                MPI_Recv(myName.data(), myNameLength, MPI_CHAR, MPI_ROOT_RANK, tag_2, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }



            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Print your name
            mpiPrintf(MPI_ALL_RANKS, "Rank %d: %s (%d characters).\n",
                      mpiGetCommRank(MPI_COMM_WORLD), myName.c_str(), std::size(myName));

            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        }

//--------------------------------------------------------------------------------------------------------------------//
//                               Example 3 - Broadcast your name using ISend and Probe + Get_cout + Recv.             //
//--------------------------------------------------------------------------------------------------------------------//
        case 3: // Example 3 - Broadcast your name using 1 ISend and Get_count + Recv
        {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 3 - Broadcast your name to all ranks using Isend and        \n" \
                               "             Probe + Get_count + Recv \n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            // Define string and its length
            std::string myName{};
            int myNameLength{};

            // Enter your name at rank 0:
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                myName = studentLogin;
                myNameLength = std::size(myName);
            }

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                             Enter your code here                                             //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // 1. Define a tag for string exchange.
            int tag_1 = 1;
            int tag_2 = 2;


            // 2. Root rank sends the message using Isend. The requests are stored in req[process count] array.
            //    Since the root doesn't send a message to itself, its request is set to MPI_REQUEST_NULL.
            //    After all messages has been sent, call MPI_Waitall.

            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                MPI_Request req[mpiGetCommSize(MPI_COMM_WORLD) - 1]; // Correct array size
                for (int i = 1; i < mpiGetCommSize(MPI_COMM_WORLD); i++) {
                    MPI_Isend(myName.data(), myNameLength, MPI_CHAR, i, tag_2, MPI_COMM_WORLD, &req[i - 1]); // Correct indexing
                }
                MPI_Waitall(mpiGetCommSize(MPI_COMM_WORLD) - 1, req, MPI_STATUSES_IGNORE); // Correct count
            }

            if (mpiGetCommRank(MPI_COMM_WORLD) != MPI_ROOT_RANK) {
                MPI_Status status;
                MPI_Probe(MPI_ROOT_RANK, tag_2, MPI_COMM_WORLD, &status);
                int myNameLength; // Declare myNameLength here
                MPI_Get_count(&status, MPI_CHAR, &myNameLength); // CORRECT: Use MPI_CHAR
                printf("Rank %d: Received name length: %d\n", mpiGetCommRank(MPI_COMM_WORLD), myNameLength); // More informative print
                myName.resize(myNameLength);
                MPI_Recv(myName.data(), myNameLength, MPI_CHAR, MPI_ROOT_RANK, tag_2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Rank %d: Received name: %s\n", mpiGetCommRank(MPI_COMM_WORLD), myName.c_str()); // Print the received name
            }

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Print your name
            mpiPrintf(MPI_ALL_RANKS, "Rank %d: %s (%d characters).\n",
                      mpiGetCommRank(MPI_COMM_WORLD), myName.c_str(), std::size(myName));


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

