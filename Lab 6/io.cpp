/**
 * @file      io.cpp
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
 * @brief     PC lab 6 / MPI-IO
 *
 * @version   2024
 *
 * @date      29 March     2020, 13:05 (created) \n
 * @date      28 March     2023, 10:15 (created) \n
 * @date      20 March     2024, 10:00 (revised) \n
 *
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <chrono>
#include <thread>
#include <sstream>
#include <string>
#include <string_view>
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
int mpiGetCommRank(MPI_Comm comm);

/// Return size of the MPI communicator.
int mpiGetCommSize(MPI_Comm comm);



//--------------------------------------------------------------------------------------------------------------------//
//                                                 Helper functions                                                   //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * C printf routine with selection which rank prints.
 * @tparam Args  - variable number of arguments.
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
 * @param argc
 * @param argv
 * @return test id
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

    MPI_Comm_rank(comm, &rank);

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

    MPI_Comm_size(comm, &size);

    return size;
}// end of mpiGetCommSize
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialize matrix.
 * @param [in, out] matrix - Matrix to initialize.
 * @param [in]      nRows  - Number of rows.
 * @param [in]      nCols  - Number of cols.
 */
void initMatrix(int *matrix,
                int nRows,
                int nCols) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            matrix[i * nCols + j] = 100 * i + j;
        }
    }
}// end of initMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Clear matrix.
 * @param [in, out] matrix - Matrix to initialize.
 * @param [in]      nRows  - Number of rows.
 * @param [in]      nCols  - Number of cols.
 */
void clearMatrix(int *matrix,
                 int nRows,
                 int nCols) {
    std::fill_n(matrix, nRows * nCols, 0);
}// end of initMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print block of the matrix.
 * @param block     - Block to print out
 * @param blockSize - Number of rows in the block
 * @param nCols     - Number of cols.
 */
void printMatrix(int *matrix,
                 int nRows,
                 int nCols) {
    static constexpr std::size_t maxTmpLen{64};

    const int rank = mpiGetCommRank(MPI_COMM_WORLD);

    std::string str{};
    char val[maxTmpLen]{};

    for (int i = 0; i < nRows; i++) {
        str.clear();

        for (int j = 0; j < nCols; j++) {
            std::sprintf(val, "%s%8d", (j == 0) ? "" : ", ", matrix[i * nCols + j]);
            str += val;
        }

        std::printf(" - Rank %2d, row %2d = [%s]\n", rank, i, str.c_str());
    }
}// end of printBlock
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print block of the matrix.
 * @param block     - Block to print out
 * @param blockSize - Number of rows in the block
 * @param nCols     - Number of cols.
 */
void printMatrix(double *matrix,
                 int nRows,
                 int nCols) {
    static constexpr std::size_t maxTmpLen{64};

    const int rank = mpiGetCommRank(MPI_COMM_WORLD);

    std::string str{};
    char val[maxTmpLen]{};

    for (int i = 0; i < nRows; i++) {
        str.clear();

        for (int j = 0; j < nCols; j++) {
            std::sprintf(val, "%s%6.3f", (j == 0) ? "" : ", ", matrix[i * nCols + j]);
            str += val;
        }

        std::printf(" - Rank %2d, row %2d = [%s]\n", rank, i, str.c_str());
    }
}// end of printBlock
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
    mpiPrintf(MPI_ROOT_RANK, "                            PPP Lab 6                                \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // Parse parameters
    const int testId = parseParameters(argc, argv);

    // Select test
    switch (testId) {
//--------------------------------------------------------------------------------------------------------------------//
//                            Example 1 - Create a file and write Hello from the root rank                            //
//--------------------------------------------------------------------------------------------------------------------//
        case 1: {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 1 - Create a file and write Hello from the root rank        \n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                            Enter your code here                                              //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Write text Hello into the MPI file using the root rank.

            // 1. Declare an MPI file.
            MPI_File file{MPI_FILE_NULL};

            // 2. Open a file called "File1.txt" with write permission.
            MPI_File_open(MPI_COMM_WORLD, "File1.txt", MPI_MODE_WRONLY, MPI_INFO_NULL, &file);


            // 3. Use a ROOT rank and write Hello into the file.
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                constexpr std::string_view msg{"Hello from rank #0\n"};

                MPI_File_write(file, msg.data(), static_cast<int>(msg.size()), MPI_CHAR, MPI_STATUS_IGNORE);
            }

            // 4. Close the file.
            MPI_File_close(&file);


            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Task finished successfully!\n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        } //case 1

//--------------------------------------------------------------------------------------------------------------------//
//                       Example 2 - Write ranks and comm size in the file in an ascending order                      //
//--------------------------------------------------------------------------------------------------------------------//
        case 2: {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 2 - Write ranks and comm size in the file in ascending order\n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                            Enter your code here                                              //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Write the rank and comm size into the file in an ascending order. Use INDIVIDUAL MPI writes.

            // 1. Declare an MPI file.
            MPI_File file{MPI_FILE_NULL};

            // 2. Open a file called "File2.txt" with write permission.
            MPI_File_open(MPI_COMM_WORLD, "File2.txt", MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

            // Declare a string and write "I am %d from %d\n"
            constexpr std::size_t maxStrSize{128};

            std::array<char, maxStrSize> str{};
            const int strSize = std::snprintf(str.data(), maxStrSize, "I am %d from %d\n",
                                              mpiGetCommRank(MPI_COMM_WORLD),
                                              mpiGetCommSize(MPI_COMM_WORLD));

            // 3. Find positions where to write into the file. Use the Exscan operation.

            int offset{};
            MPI_Exscan((void *) &strSize, (void *) &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


            // 4. Use an individual mpi write call at an appropriate position.
            //    Check the performance and then try to use a collective operation instead.

            MPI_File_write_at(file, offset, str.data(), strSize, MPI_CHAR, MPI_STATUS_IGNORE);


            // 5. Close the file.

            MPI_File_close(&file);
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Task finished successfully!\n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        } //case 2

//--------------------------------------------------------------------------------------------------------------------//
//                                  Example 3 - Write a matrix distributed over rows                                  //
//--------------------------------------------------------------------------------------------------------------------//
        case 3: {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 3 - Write a matrix distributed over rows                    \n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            // File name
            constexpr std::string_view filename{"File3.dat"};

            // 16 x 4 matrix
            constexpr int nRows = 16;
            constexpr int nCols = 4;

            // Distribution
            const int lRows = nRows / mpiGetCommSize(MPI_COMM_WORLD);

            // Global matrix in the root.
            std::vector<int> gMatrix{};
            // Local stripe on each rank.
            std::vector<int> lMatrix{};

            // Initialize matrix on the root.
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                // If the file exists, delete it.
                MPI_File_delete(filename.data(), MPI_INFO_NULL);

                gMatrix.resize(nRows * nCols);

                initMatrix(gMatrix.data(), nRows, nCols);

                std::printf("Original array:\n");
                printMatrix(gMatrix.data(), nRows, nCols);
                std::putchar('\n');
            }

            lMatrix.resize(lRows * nCols);

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                            Enter your code here                                              //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Create a matrix distributed over row blocks into a binary file. Use a collective MPI-IO.

            // 1. Create a datatype for Matrix row.

            MPI_Datatype rowType{MPI_DATATYPE_NULL};
            MPI_Type_contiguous(nCols, MPI_INT, &rowType);
            MPI_Type_commit(&rowType);

            // 2. Scatter gMatrix into lMatrix.

            MPI_Scatter(gMatrix.data(), lRows, rowType, lMatrix.data(), lRows * nCols, MPI_INT, MPI_ROOT_RANK,
                        MPI_COMM_WORLD);

            // 3. Declare an MPI file.

            MPI_File file{MPI_FILE_NULL};

            // 4. Open a file with "filename" with write permission.

            MPI_File_open(MPI_COMM_WORLD, filename.data(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);

            // 5. Set a file view. Hint - use displacement to move at appropriate position in the file.

            MPI_File_set_view(file, lRows * nCols * mpiGetCommRank(MPI_COMM_WORLD) * sizeof(int), MPI_INT, rowType,
                              "native", MPI_INFO_NULL);

            // 6. Use a collective write.

            MPI_File_write_all(file, lMatrix.data(), lRows * nCols, MPI_INT, MPI_STATUS_IGNORE);

            // 7. Close the file.
            MPI_File_close(&file);

            // 8. Delete the matrix row datatype.

            MPI_Type_free(&rowType);


            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // read file and print it out.
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                std::fill(lMatrix.begin(), lMatrix.end(), 0);

                std::ifstream file(filename.data(), std::ios::binary);

                file.read(reinterpret_cast<char *>(gMatrix.data()), nRows * nCols * sizeof(int));

                std::printf("File %s:\n", filename.data());
                printMatrix(gMatrix.data(), nRows, nCols);
            }

            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        } //case 3

//--------------------------------------------------------------------------------------------------------------------//
//                             Example 4 - Read a PGM picture in tiles and make it lighter                            //
//--------------------------------------------------------------------------------------------------------------------//
        case 4: {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 4 - Read a PGM picture in tiles and make it lighter \n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            /// File header structure, POD.
            struct Header {
                /// Default constructor.
                Header() = default;

                /// Create header by parsing header string.
                Header(std::string headerString) {
                    char line[100];
                    std::stringstream ss(headerString);

                    // Ignore first line
                    ss.getline(line, 100);
                    // Ignore second line
                    ss.getline(line, 100);
                    // Read dimensions
                    ss >> nRows;
                    ss >> nCols;

                    // Ignore last line
                    ss.getline(line, 100);
                    size = int(ss.tellg()) + 4; // + 4 end of line chars
                }

                /// Return header as a string
                std::string toString() {
                    constexpr std::size_t maxBufferSize{500};

                    std::array<char, maxBufferSize> buffer{};

                    std::snprintf(buffer.data(), maxBufferSize, "P5\n# Created by IrfanView\n%d %d\n255\n", nRows,
                                  nCols);
                    return std::string{buffer.data()};
                }

                /// Number of rows
                int nRows{};
                /// Number of cols
                int nCols{};
                /// Size of the header string
                int size{};
            };

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                        Enter your code here                                                  //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Read an input file distributed over tiles among ranks. First, you have to parse the header by the root.
            // Run a gamma correction kernel
            // Store the new image. The root rank writes the header back into the file.

            // File names
            constexpr std::string_view inFilename{"../lena_in.pgm"};
            constexpr std::string_view outFilename{"lena_out.pgm"};

            // Input and output file.
            MPI_File inFile{MPI_FILE_NULL};
            MPI_File outFile{MPI_FILE_NULL};

            // 1. Open input file.

            MPI_File_open(MPI_COMM_WORLD, inFilename.data(), MPI_MODE_RDONLY, MPI_INFO_NULL, &inFile);

            // 2. Delete the output file and the create a new one.

            MPI_File_delete(outFilename.data(), MPI_INFO_NULL);

            // Read Header by the root rank, expect maximum size 500 chars and write the same one to the output file.
            Header fileHeader{};

            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                constexpr std::size_t maxBufferSize{500};

                std::array<char, maxBufferSize> buffer{};

                // 3. Read the header string from the input file.
                MPI_File_read_at(inFile, 0, buffer.data(), maxBufferSize, MPI_CHAR, MPI_STATUS_IGNORE);

                fileHeader = Header(buffer.data());

                // 4. Write the header into the output file, use fileHeader.size and fileHeader.toString().c_str().

                MPI_File_open(MPI_COMM_WORLD, outFilename.data(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL,
                              &outFile);
            }

            // Header can be sent as a contiguous block.
            static_assert(sizeof(Header) == 3 * sizeof(int));

            // 5. Broadcast the header structure. Header is POD, can be sent as a contiguous block.

            int sendBuff[] = {fileHeader.nRows, fileHeader.nCols, fileHeader.size};

            MPI_Bcast(&sendBuff, 3, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

            // Get number of local rows and cols.
            const int lRows = fileHeader.nRows / static_cast<int>(std::sqrt(mpiGetCommSize(MPI_COMM_WORLD)));
            const int lCols = fileHeader.nCols / static_cast<int>(std::sqrt(mpiGetCommSize(MPI_COMM_WORLD)));

            // 6. Create a subarray to read/write a tile from/to the input/output file.


            // Create a datatype for the tile.
            MPI_Datatype tileType{MPI_DATATYPE_NULL};
            MPI_Type_create_subarray(2, &fileHeader.nRows, &fileHeader.nCols, &lRows, &lCols,
                                      MPI_INT, MPI_ORDER_C, &tileType);




            std::vector<unsigned char> image(lRows * lCols);

            // 7.  Set file view in the input and output files.



            mpiPrintf(MPI_ROOT_RANK, " Reading input file...\n");
            // 8. Read input image.


            // Run gamma correction kernel.
            mpiPrintf(MPI_ROOT_RANK, " Running gamma correction ...\n");
            std::for_each(image.begin(), image.end(), [](auto &pixel) {
                double correctedPixel = 255.0 * std::pow(pixel / 255.0, 1.0 / 2.2);

                pixel = static_cast<unsigned char>(std::clamp(correctedPixel, 0.0, 255.0));
            });

            mpiPrintf(MPI_ROOT_RANK, " Writing final image ...\n");

            // 8. Write final image.


            // 9. Free created datatypes.



            // 10. Close input and output file.



            mpiPrintf(MPI_ROOT_RANK, " Done\n");
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        } //case 4

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
