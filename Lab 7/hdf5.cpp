/**
 * @file      hdf5.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 7 / HDF5
 *
 * @version   2023
 *
 * @date      05 April     2020, 12:10 (created) \n
 * @date      04 April     2023, 12:22 (created) \n
 *
 */

#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <string>
#include <cmath>
#include <vector>
#include <sstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cstdarg>
#include <array>


#include <mpi.h>
#include <hdf5.h>


/// constant used for mpi printf
constexpr int MPI_ALL_RANKS = -1;
/// constant for root rank
constexpr int MPI_ROOT_RANK = 0;

//--------------------------------------------------------------------------------------------------------------------//
//                                            Helper function prototypes                                              //
//--------------------------------------------------------------------------------------------------------------------//
/// Wrapper to the C printf routine and specifies who can print (a given rank or all).
void mpiPrintf(int who, const char *__restrict__ format, ...);

/// Flush standard output.
void mpiFlush();

/// Parse command line parameters.
int parseParameters(int argc, char **argv);

/// Return MPI rank in a given communicator.
int mpiGetCommRank(const MPI_Comm &comm);

/// Return size of the MPI communicator.
int mpiGetCommSize(const MPI_Comm &comm);

/// Execute a given command in the shell and return the output.
std::string exec(const char *cmd);

//--------------------------------------------------------------------------------------------------------------------//
//                                                 Helper functions                                                   //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * C printf routine with selection which rank prints.
 * @param who    - which rank should print. If -1 then all prints.
 * @param format - format string.
 * @param ...    - other parameters.
 */
void mpiPrintf(int who,
               const char *__restrict__ format,
               ...) {
    if ((who == MPI_ALL_RANKS) || (who == mpiGetCommRank(MPI_COMM_WORLD))) {
        va_list args;
        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end(args);
    }
}// end of mpiPrintf
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush stdout and call barrier to prevent message mixture.
 */
void mpiFlush() {
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    // A known hack to correctly order writes
    usleep(100);
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

    return atoi(argv[1]);
}// end of parseParameters
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get MPI rank within the communicator.
 * @param [in] comm - actual communicator.
 * @return rank within the comm.
 */
int mpiGetCommRank(const MPI_Comm &comm) {
    int rank = MPI_UNDEFINED;

    MPI_Comm_rank(comm, &rank);

    return rank;
}// end of mpiGetCommRank
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get size of the communicator.
 * @param [in] comm - actual communicator.
 * @return number of ranks within the comm.
 */
int mpiGetCommSize(const MPI_Comm &comm) {
    int size = -1;

    MPI_Comm_size(comm, &size);

    return size;
}// end of mpiGetCommSize
//----------------------------------------------------------------------------------------------------------------------

/**
 * Execute command in shell.
 * @param [in] command line
 * @return Standard output form the commandline.
 */
std::string exec(const char *cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr < FILE, decltype(&pclose) > pipe(popen(cmd, "r"), pclose);

    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    // Read the commnadline output
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}// end of exec
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialize matrix.
 * @tparam T               - Datatype of the matrix.
 * @param [in, out] matrix - Matrix to initialize.
 * @param [in]      nRows  - Number of rows.
 * @param [in]      nCols  - Number of cols.
 */
template<typename T>
void initMatrix(T *matrix,
                int nRows,
                int nCols) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            matrix[i * nCols + j] = T(100 * i + j);
        }
    }
}// end of initMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Clear matrix.
 * @tparam T               - Datatype of the matrix.
 * @param [in, out] matrix - Matrix to initialize.
 * @param [in]      nRows  - Number of rows.
 * @param [in]      nCols  - Number of cols.
 */
template<typename T>
void clearMatrix(T *matrix,
                 int nRows,
                 int nCols) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            matrix[i * nCols + j] = T(0);
        }
    }
}// end of initMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print block of the matrix
 * @tparam T         - Datatype of the matrix.
 * @param  [in, out] matrix - Block to print out.
 * @param  [in]      nRows  - Number of rows in the block.
 * @param  [in]      nCols  - Number of cols.
 */
template<typename T>
void printMatrix(T *matrix,
                 int nRows,
                 int nCols) {
    std::string str;
    char val[16];

    std::string valueFormat = (std::is_same<T, double>() || std::is_same<T, float>()) ? "%6.3f" : "%8d";

    for (int i = 0; i < nRows; i++) {
        str = "";
        sprintf(val, " - Rank %2d = [", mpiGetCommRank(MPI_COMM_WORLD));
        str += val;

        for (int j = 0; j < nCols - 1; j++) {
            sprintf(val, (valueFormat + ", ").c_str(), matrix[i * nCols + j]);
            str += val;
        }
        sprintf(val, (" " + valueFormat + "]\n").c_str(), matrix[i * nCols + nCols - 1]);
        str += val;
        printf("%s", str.c_str());
    }
}// end of printMatrix
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//                                                   Main routine                                                     //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Main function.
 */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // Set stdout to print out
    setvbuf(stdout, NULL, _IONBF, 0);

    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiPrintf(MPI_ROOT_RANK, "                            PPP Lab 7                                \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // Parse parameters
    const int testId = parseParameters(argc, argv);
    // Select test
    switch (testId) {
//--------------------------------------------------------------------------------------------------------------------//
//                        Example 1 - Create a HDF5 file and write a scalar from the root rank                        //
//--------------------------------------------------------------------------------------------------------------------//
        case 1: {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 1 - Create a HDF5 file and write a scalar from the root rank\n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                            Enter your code here                                              //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // The first example only uses the root rank (serial IO). We first create a HDF5 file, then create a dataset and
            // finally write a scalar value of 128 into it.

            const char *fileName = "File1.h5";
            const char *datasetName = "Dataset-1";

            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                // 1. Declare an HDF5 file.

                hid_t file;

                // 2. Create a file with write permission. Use such a flag that overrides existing file.
                //    The list of flags is in the header file called H5Fpublic.h
                mpiPrintf(MPI_ROOT_RANK, " Creating file... \n");


                file = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

                // 3. Create file and memory spaces. We will only write a single value.

                const hsize_t rank = 1;
                const hsize_t size = 1;

                hid_t filespace = H5Screate_simple(rank, &size, NULL);
                hid_t memspace = H5Screate_simple(rank, &size, NULL);

                // 4. Create a dataset of a size [1] and int datatype.
                //    The list of predefined datatypes can be found in H5Tpublic.h


                auto dataset = H5Dcreate(file,
                                         datasetName,
                                         H5T_NATIVE_INT,
                                         filespace,
                                         H5P_DEFAULT,
                                         H5P_DEFAULT,
                                         H5P_DEFAULT);

                //5. Write value into the dataset.
                mpiPrintf(MPI_ROOT_RANK, " Writing scalar value... \n");
                const int value = 128;

                H5Dwrite(dataset, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, &value);

                // 6. Close dataset.

                H5Dclose(dataset);;
                H5Fclose(file);


                // 7. Close file
                mpiPrintf(MPI_ROOT_RANK, " Closing file... \n");

            }

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // 8. Use command line tools h5ls and h5dump to see what's in the file.
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
                mpiPrintf(MPI_ROOT_RANK, "h5ls output: \n");

                char cmd[256] = {0};

                sprintf(cmd, "h5ls -f -r  %s", fileName);
                mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd).c_str());
                mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");


                mpiPrintf(MPI_ROOT_RANK, "h5dump output: \n");
                sprintf(cmd, "h5dump -p -d %s  %s", datasetName, fileName);

                mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd).c_str());
            }

            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        } //case 1


//--------------------------------------------------------------------------------------------------------------------//
//                                  Example 2 - Write a matrix distributed over rows                                  //
//--------------------------------------------------------------------------------------------------------------------//
        case 2: {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 2 - Write a matrix distributed over rows                    \n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            // File name and dataset name
            const char *filename = "File2.h5";
            const char *datasetname = "matrix";

            // 16 x 4 matrix
            constexpr int nRows = 16;
            constexpr int nCols = 4;

            // Distribution
            const int lRows = nRows / mpiGetCommSize(MPI_COMM_WORLD);

            // global matrix in the root
            int *gMatrix = nullptr;
            // local stripe on each rank
            int *lMatrix = nullptr;

            // Initialize matrix on the root
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                gMatrix = new int[nRows * nCols];

                initMatrix<int>(gMatrix, nRows, nCols);

                printf("Original array:\n");
                printMatrix(gMatrix, nRows, nCols);
                printf("\n");
            }

            lMatrix = new int[lRows * nCols];

            MPI_Datatype MPI_ROW;
            MPI_Type_contiguous(nCols, MPI_INT, &MPI_ROW);
            MPI_Type_commit(&MPI_ROW);

            // Scatter matrix over rows
            MPI_Scatter(gMatrix, lRows, MPI_ROW, lMatrix, lRows, MPI_ROW, 0, MPI_COMM_WORLD);

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                            Enter your code here                                              //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Let's have a matrix nRows * nCols scattered by rows onto ranks. Each rank maintains lRows * nCols
            // The goal is to create a dataset in the HDF5 file and write the matrix using collective IO there.

            // 1. Declare an HDF5 file.

            hid_t file;

            // 2. Create a property list to open the file using MPI-IO in the MPI_COMM_WORLD communicator.

            hid_t accelist = H5Pcreate(H5P_DATASET_ACCESS);
            file = H5Create(filename, H5F_COMM, MPI_INFO_NULL, &file); // todo

            // 3. Create a file called (filename) with write permission. Use such a flag that overrides existing file.
            //    The list of flags is in the header file called H5Fpublic.h
            mpiPrintf(MPI_ROOT_RANK, " Creating file... \n");

            file = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            // 4. Close file access list.

            // skippp

            // 5. Create file space - a 2D matrix [nRows][nCols]
            //    Create mem space  - a 2D matrix [lRows][nCols] mapped on 1D array lMatrix.

            const hsize_t rank = 2;
            const hsize_t size[2] = {nRows, nCols};
            const hsize_t size2[2] = {nCols, lRows};

            hid_t filespace = H5Screate_simple(rank, &size, NULL);
            hid_t memspace = H5Screate_simple(rank, &size2, NULL);


            // 6. Create a dataset. The name is store in datasetname, datatype is int. All other parameters are default.
            mpiPrintf(MPI_ROOT_RANK, " Creating dataset... \n");


            auto dataset = H5Dcreate(file,
                                     datasetName,
                                     H5T_NATIVE_INT,
                                     filespace,
                                     H5P_DEFAULT,
                                     H5P_DEFAULT,
                                     H5P_DEFAULT);


            // 7. Select a hyperslab to write a local submatrix into the dataset.
            mpiPrintf(MPI_ROOT_RANK, " Selecting hyperslab... \n");

            H5Sselect_hyperslab(filespace,
                                H5S_SELECT_SET,
                                &lRows, // start
                                NULL,   // stride
                                &lRows,  // count
                                NULL);  // block


            // 8. Create XFER property list and set Collective IO.

            H5Sselect_hyperslab(memspace,
                                H5S_SELECT_SET,
                                &lRows, // start
                                NULL,   // stride
                                &lRows,  // count
                                NULL);  // block

            // 9. Write data into the dataset.
            mpiPrintf(MPI_ROOT_RANK, " Writing data... \n");


            // 10. Close XREF property list.


            // 11. Close dataset.


            // 12. Close dataset.
            mpiPrintf(MPI_ROOT_RANK, " Closing file... \n");


            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            MPI_Type_free(&MPI_ROW);


            delete[] gMatrix;
            delete[] lMatrix;

            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
                mpiPrintf(MPI_ROOT_RANK, "h5ls output: \n");

                char cmd[256] = {0};

                sprintf(cmd, "h5ls -f -r  %s", filename);
                mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd).c_str());
                mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");


                mpiPrintf(MPI_ROOT_RANK, "h5dump output: \n");
                sprintf(cmd, "h5dump -p -d %s  %s", datasetname, filename);
                mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd).c_str());
            }

            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        }// case 2

//--------------------------------------------------------------------------------------------------------------------//
//                                         Example 3 - Hadamard product A°B'                                          //
//--------------------------------------------------------------------------------------------------------------------//
        case 3: {
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, " Example 3 -  Hadamard product C = A°B' with data in a HDF5 file     \n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            // Input file name
            const char *filename = "Matrix-File.h5";

            // HDF5 groups for input and output data.
            const char *inputGroupName = "Inputs";
            const char *outputGroupName = "Outputs";

            // Matrix A name
            const char *matrixAName = "Matrix-A";
            // Matrix B name
            const char *matrixBName = "Matrix-B";
            // Matrix C name
            const char *matrixCrefName = "Matrix-C-ref";


            /**
             * @struct
             * @brief 2D dimension sizes
             */
            struct Dim2D {
                /// Default constructor.
                Dim2D() : nRows(0), nCols(0) {};

                /// Initializing constructor.
                Dim2D(hsize_t rows, hsize_t cols) : nRows(rows), nCols(cols) {};

                /// Array -> Dims
                Dim2D(const hsize_t dims[]) : nRows(dims[0]), nCols(dims[1]) {};

                /// Get number of elements
                int nElements() const { return nRows * nCols; };

                /// Convert dimensions to string
                std::string toString() const {
                    char str[100];
                    sprintf(str, "[%lld, %lld]", nRows, nCols);
                    return std::string(str);
                }

                /// Convert dimensions to HDF5 array (not very nice, but acceptable for this example, const version
                const hsize_t *toArray() const {
                    return static_cast<const hsize_t *>(&nRows);
                }

                /// Convert dimensions to HDF5 array (not very nice, but acceptable for this example
                hsize_t *toArray() {
                    return static_cast<hsize_t *>(&nRows);
                }

                /// Number of rows.
                hsize_t nRows;
                /// Number of columns,
                hsize_t nCols;
            };

            // Print content of the matrix
            if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK) {
                mpiPrintf(MPI_ROOT_RANK, "h5ls output: \n");

                char cmd[256] = {0};

                // List the content of the file
                sprintf(cmd, "h5ls -f -r  %s", filename);
                mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd).c_str());
                mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");


                //#define DEBUG
                // If DEBUG then print both header and data.
#ifdef DEBUG
                mpiPrintf(MPI_ROOT_RANK, "h5dump output: \n");
                sprintf(cmd, "h5dump %s ", filename);
#else
                mpiPrintf(MPI_ROOT_RANK, "h5dump output (dataset data omitted, enable by using -d parameter): \n");
                sprintf(cmd, "h5dump -H %s ", filename);
#endif
                mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd).c_str());
                mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");

            }

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                                        Enter your code here                                                  //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // The goal of this exercise is to read two matrices from the input file and calculate the Hadamard product
            // of A and B transposed C = A ° B'. Hadamard product is a simple element wise multiplication.
            //  C = A ° B' ; for each i, j: c[i][j] = a[i][j] * b[j][i].
            //
            // To make it fast, we will distribute matrices A and C by row blocks, while the matrix B by column blocks.
            // The file also contains a reference output we can compare to.


            // Local part of matrices you need.
            double *matrixA = nullptr;
            double *matrixB = nullptr;
            // Matrix to store my result.
            double *matrixC = nullptr;
            // Reference matrix in the file.
            double *matrixCref = nullptr;


            // 1. Create a property list to open the HDF5 file using MPI-IO in the MPI_COMM_WORLD communicator.


            // 2. Open a file called (filename) with read-only permission.
            //    The list of flags is in the header file called H5Fpublic.h
            mpiPrintf(MPI_ROOT_RANK, " Opening file... \n");


            // 3. Close file access list.



            // 4. Open HDF5 groups with input and output matrices.


            // 5. Open HDF5 datasets with input and output matrices


            // 6. Write a lambda function to read the dataset size. The routine takes dataset ID and returns Dims2D.

            //   i.  Get the dataspace from the dataset using H5Dget_space.
            //   ii. Read dataset sizes using H5Sget_simple_extent_dims.
            //       The rank of the dataspace is 2 (2D matrices) and the dimensions are stored in Row, Col manner (row major)
            //       You can use a conversion dims.toArray() to pass the structure as an array.
            auto getDims = [](hid_t dataset) -> Dim2D {
                Dim2D dims;

                return dims;
            };// end of getDims


            // 7. Get global matrix dimension sizes.
            mpiPrintf(MPI_ROOT_RANK, " Reading dimension sizes... \n");
            Dim2D gDimsA;
            Dim2D gDimsB;
            Dim2D gDimsC;

            // 8. Calculate local matrix dimension sizes. A, C and Cref are distributed by rows, B by columns.
            Dim2D lDimsA;
            Dim2D lDimsB;
            Dim2D lDimsC;

            // Print out dimension sizes
            mpiPrintf(MPI_ROOT_RANK, "  - Number of ranks: %d\n", mpiGetCommSize(MPI_COMM_WORLD));
            mpiPrintf(MPI_ROOT_RANK, "  - Matrix A global and local size %s / %s \n",
                      gDimsA.toString().c_str(), lDimsA.toString().c_str());
            mpiPrintf(MPI_ROOT_RANK, "  - Matrix B global and local size %s / %s \n",
                      gDimsB.toString().c_str(), lDimsB.toString().c_str());
            mpiPrintf(MPI_ROOT_RANK, "  - Matrix C global and local size %s / %s \n",
                      gDimsC.toString().c_str(), lDimsC.toString().c_str());


            // Allocate memory for local arrays
            matrixA = new double[lDimsA.nElements()];
            matrixB = new double[lDimsB.nElements()];
            matrixC = new double[lDimsC.nElements()];
            matrixCref = new double[lDimsC.nElements()];


            // 9. Write a lambda function to read a particular slab at particular ranks.
            //    i.   Create a 2D filespace, then select the part of the dataset you want to read.
            //    ii.  Create a memspace where to read the data.
            //    iii. Enable collective MPI-IO.
            //    iv.  Read the slab.
            auto readSlab = [](hid_t dataset,
                               const Dim2D &slabStart, const Dim2D &slabSize, const Dim2D &datasetSize,
                               double *data) -> void {


            };// end of readSlab


            // 10. Read parts of the matrices A, B and Cref.
            mpiPrintf(MPI_ROOT_RANK, "  Reading matrix A... \n");


            mpiPrintf(MPI_ROOT_RANK, "  Reading matrix B... \n");


            mpiPrintf(MPI_ROOT_RANK, "  Reading matrix Cref... \n");


            mpiPrintf(MPI_ROOT_RANK, "  Calculating C = A ° B' ... \n");

            // Calculate the Hadamard product C = A ° B'
            clearMatrix(matrixC, lDimsC.nRows, lDimsC.nCols);
            for (int row = 0; row < lDimsC.nRows; row++) {
                for (int col = 0; col < lDimsC.nCols; col++) {
                    matrixC[row * lDimsC.nCols + col] =
                            matrixA[row * lDimsA.nCols + col] * matrixB[col * lDimsB.nCols + row];
                }
            }


            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef DEBUG
            mpiPrintf(MPI_ROOT_RANK, " matrixA [%ld, %ld]\n", lDimsA.nRows, lDimsA.nCols);
            printMatrix(matrixA, lDimsA.nRows, lDimsA.nCols);
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();


            mpiPrintf(MPI_ROOT_RANK, " matrixB [%ld, %ld]\n", lDimsB.nRows, lDimsB.nCols);
            printMatrix(matrixB, lDimsB.nRows, lDimsB.nCols);
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            mpiPrintf(MPI_ROOT_RANK, " matrixC [%ld, %ld]\n", lDimsC.nRows, lDimsC.nCols);
            printMatrix(matrixC, lDimsC.nRows, lDimsC.nCols);
            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
#endif

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            mpiPrintf(MPI_ROOT_RANK, "  Verification C == Cref ... \n");

            // Validate data C == Cref and print out maximum absolute error.
            double maxError = 0;
            double globalError = 0;
            for (int row = 0; row < lDimsC.nRows; row++) {
                for (int col = 0; col < lDimsC.nCols; col++) {
                    maxError = std::max(maxError,
                                        fabs(matrixCref[row * lDimsC.nCols + col] - matrixC[row * lDimsC.nCols + col]));
                }
            }

            MPI_Reduce(&maxError, &globalError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            mpiPrintf(MPI_ROOT_RANK, "  Maximum abs error = %e \n", globalError);


            // 11. Close datasets

            // 12. Close HDF5 groups

            // 13. Close HDF5 file



            // Delete matrices
            delete[] matrixA;
            delete[] matrixB;
            delete[] matrixC;
            delete[] matrixCref;

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


            mpiFlush();
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();
            break;
        }// case 3


        default: {
            mpiPrintf(0, " !!!                     Unknown test number                     !!!\n");
            mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
            mpiFlush();

            break;
        }
    }// switch


    MPI_Finalize();
    return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------



