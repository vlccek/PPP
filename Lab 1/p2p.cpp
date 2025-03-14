/**
 * @file      p2p.cpp
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
 * @brief     PC lab 1 / MPI point-to-point communications
 *
 * @version   2023
 *
 * @date      12 February  2020, 17:21 (created) \n
 * @date      21 February  2023, 11:52 (updated) \n
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

/// Constant used for mpi printf.
constexpr int MPI_ALL_RANKS = -1;
/// Constant for root rank.
constexpr int MPI_ROOT_RANK = 0;

//--------------------------------------------------------------------------------------------------------------------//
//                                               Function prototypes                                                  //
//--------------------------------------------------------------------------------------------------------------------//
/// Wrapper to the C printf routine and specifies who can print (a given rank or all).
template<typename... Args>
void mpiPrintf(int who, const std::string_view format, Args... args);
/// Flush standard output.
void mpiFlush();
/// Parse command line parameters.
int  parseParameters(int argc, char** argv);

//--------------------------------------------------------------------------------------------------------------------//
//                                            Function to be implemented                                              //
//--------------------------------------------------------------------------------------------------------------------//

/// Return MPI rank in a given communicator.
int mpiGetCommRank(MPI_Comm comm);
/// Return size of the MPI communicator.
int mpiGetCommSize(MPI_Comm comm);

/// Exchange the message with the dst rank using blocking MPI communications.
void mpiBlockingExchange(int* message, int size, int dst);
/// Exchange messages between this rank and the destination using non-blocking MPI communications.
void mpiNonBlockingExchange(const int* messageSend, int* messageRecv, int size, int dst);

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
void mpiPrintf(int who, const std::string_view format, Args... args)
{
  if ((who == MPI_ALL_RANKS) || (who == mpiGetCommRank(MPI_COMM_WORLD)))
  {
    if constexpr (sizeof...(args) == 0)
    {
      std::printf("%s", std::data(format));
    }
    else
    {
      std::printf(std::data(format), args...);
    }
  }
}// end of mpiPrintf
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush stdout and call barrier to prevent message mixture.
 */
void mpiFlush()
{
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
int parseParameters(int argc, char** argv)
{
  if (argc != 2)
  {
    mpiPrintf(MPI_ROOT_RANK, "!!!                  Please specify test number!               !!!\n");
    mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  return std::stoi(argv[1]);
}// end of parseParameters
//----------------------------------------------------------------------------------------------------------------------



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                          Implement following functions                                             //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Get MPI rank within the communicator.
 * @param [in] comm - actual communicator.
 * @return rank within the comm.
 */
int mpiGetCommRank(MPI_Comm comm)
{
  int rank{};



  return rank;
}// end of mpiGetCommRank
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get size of the communicator.
 * @param [in] comm - actual communicator.
 * @return number of ranks within the comm.
 */
int mpiGetCommSize(MPI_Comm comm)
{
  int size{};



  return size;
}// end of mpiGetCommSize
//----------------------------------------------------------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                         Blocking exchange of two messages                                          //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Mutual message exchange between two partners using blocking exchange.
 * @param [out, in] message - message to be exchanged.
 * @param [in]      size    - size of the message.
 * @param [in]      dst     - destination rank.
 */
void mpiBlockingExchange(int* message,
                         int  size,
                         int  dst)
{
  MPI_Send(message, size, MPI_INT, dst, 0, MPI_COMM_WORLD);
}// end of mpiBlockingExchange
//----------------------------------------------------------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                        Nonblocking exchange of two messages                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Mutual message exchange between two partners.
 * @param [out, in] messageSend - message to be sent.
 * @param [out, in] messageRecv - message to be received.
 * @param [in]      size        - size of the message.
 * @param [in]      dst         - destination rank.
 */
void mpiNonBlockingExchange(const int* messageSend,
                            int*       messageRecv,
                            int        size,
                            int        dst)
{
  
}// end of mpiNonBlockingExchange
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//                                                   Main routine                                                     //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Main function
 */
int main(int argc, char** argv)
{
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                            Initialize MPI correctly                                              //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  MPI_Init(&argc, &argv);

  // Disable buffering, ignore the return value
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
  mpiPrintf(MPI_ROOT_RANK, "                          PPP Lab 1                               \n");
  mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
  mpiFlush();

  // Parse parameters
  const int testId = parseParameters(argc, argv);

  // Select test
  switch (testId)
  {
    case 1:  // example 1 - Hello world
    {
      mpiFlush();
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, " Example 1 - Hello world                                          \n");
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      mpiPrintf(MPI_ALL_RANKS, " My rank is %d / %d\n",
                               mpiGetCommRank(MPI_COMM_WORLD),
                               mpiGetCommSize(MPI_COMM_WORLD));
      break;

    }// case 1

    case 2: // example 2 - two messages blocking
    {
      mpiFlush();
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, " Example 2 - Exchange two messages                                \n");
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      const int srcRank    =  mpiGetCommRank(MPI_COMM_WORLD) % 2;
      const int dstRank    = (mpiGetCommRank(MPI_COMM_WORLD) + 1) % 2;
      const int size       = 1;
      
      int message[1] = {srcRank};

      mpiPrintf(srcRank, " Sending message:   %d->%d [%d]\n", srcRank, dstRank, message[0]);

      mpiBlockingExchange(message, size, dstRank);

      mpiPrintf(srcRank, " Receiving message: %d->%d [%d]\n", dstRank, srcRank, message[0]);

      mpiFlush();
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();
      break;
    }// case 2

    case 3: // example 3 - deadlock test blocking
    {
      mpiFlush();
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, " Example 3 - Deadlock test - blocking                             \n");
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      constexpr int MAX_SIZE = 32 * 1024 * 1024; // 64 MB

      const int srcRank =  mpiGetCommRank(MPI_COMM_WORLD) % 2;
      const int dstRank = (mpiGetCommRank(MPI_COMM_WORLD) + 1) % 2;

      auto message = std::make_unique<int[]>(MAX_SIZE);

      for (int size = 1; size < MAX_SIZE ; size *= 2)
      {
        mpiPrintf(MPI_ROOT_RANK, " Sending %dB ...", size * 4);
        mpiBlockingExchange(message.get(), size, dstRank);
        mpiPrintf(MPI_ROOT_RANK, " Done \n");
      }

      mpiFlush();
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      break;
    }// case 3

    case 4: // example 4 - two messages non blocking
    {
      mpiFlush();
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, " Example 4 - Exchange two messages - non blocking                 \n");
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      const int srcRank        =  mpiGetCommRank(MPI_COMM_WORLD) % 2;
      const int dstRank        = (mpiGetCommRank(MPI_COMM_WORLD) + 1) % 2;
      const int messageSend[1] = {srcRank};
      const int size           = 1;
      
      int messageRecv[1] = {-1};

      mpiPrintf(srcRank, " Sending message:   %d->%d [%d]\n", srcRank, dstRank, messageSend[0]);

      mpiNonBlockingExchange(messageSend, messageRecv, size, dstRank);

      mpiPrintf(srcRank, " Receiving message: %d->%d [%d]\n", dstRank, srcRank, messageRecv[0]);

      mpiFlush();
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      break;
    }// case 4

    case 5: // example 5 - Deadlock test - non blocking
    {
      mpiFlush();
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, " Example 5 - Deadlock test - non blocking                         \n");
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      constexpr int MAX_SIZE = 32 * 1024 * 1024; // 64MB

      const int srcRank =  mpiGetCommRank(MPI_COMM_WORLD) % 2;
      const int dstRank = (mpiGetCommRank(MPI_COMM_WORLD) + 1) % 2;

      auto messageSend = std::make_unique<int[]>(MAX_SIZE);
      auto messageRecv = std::make_unique<int[]>(MAX_SIZE);

      for (int size = 1; size < MAX_SIZE ; size *= 2)
      {
        mpiPrintf(MPI_ROOT_RANK, " Sending %8dB ...", size * 4);
        mpiNonBlockingExchange(messageSend.get(), messageRecv.get(), size, dstRank);
        mpiPrintf(MPI_ROOT_RANK, " Done \n");
      }

      mpiFlush();

      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      break;
    }// case 5

    case 6: // example 5 - Deadlock test - bandwidth
    {
      mpiFlush();
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, " Example 6 - Bandwidth test - non blocking                        \n");
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      constexpr int MAX_SIZE = 32 * 1024 * 1024; // 64MB

      const int srcRank =  mpiGetCommRank(MPI_COMM_WORLD) % 2;
      const int dstRank = (mpiGetCommRank(MPI_COMM_WORLD) + 1) % 2;
      
      int nRept = MAX_SIZE;

      auto messageSend = std::make_unique<int[]>(MAX_SIZE);
      auto messageRecv = std::make_unique<int[]>(MAX_SIZE);

      // test one size
      for (int size = 1; size < MAX_SIZE ; size *= 2)
      { // repeat for MAX_REPT times
        double startTime = MPI_Wtime();

        for (int rept = 0; rept < nRept; rept++ )
        {
          mpiNonBlockingExchange(messageSend.get(), messageRecv.get(), size, dstRank);
        }

        double timePerTransfer = (MPI_Wtime() - startTime) / nRept;
        // Bidirectional bandwidth
        double bandwidth       = 2 * (4 * size) / ( 1024 * 1024 * timePerTransfer);
        mpiPrintf(MPI_ROOT_RANK, "Message size: %8dB, time: %8.2fus, bandwidth: %8.2fMB/s\n",
                                 size * 4, timePerTransfer * 1000 * 1000, bandwidth);


        nRept = nRept / 2;
      }

      mpiFlush();

      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      break;
    }// case 6


    default:
    {
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, "!!!                     Unknown test number                    !!!\n");
      mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
      mpiFlush();

      break;
    }
  }// switch

  MPI_Finalize();

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                             Finalize MPI correctly                                               //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


}// end of main
//----------------------------------------------------------------------------------------------------------------------

