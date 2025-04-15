#
# @file   CMakeLists.txt
#
# @author David Bayer \n
#         Faculty of Information Technology \n
#         Brno University of Technology \n
#         ibayer@fit.vutbr.cz
#
# @brief  CMakeLists for lab6
#
# @date   20 March 2024, 17:00 (created) \n
#

cmake_minimum_required(VERSION 3.20)

# project name
project(lab6 LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find and setup MPI package
set(MPI_CXX_SKIP_MPICXX ON)

find_package(MPI REQUIRED COMPONENTS CXX)

# copy lena_in.pgm to build directory
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lena_in.pgm" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# compile io
add_executable(io ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp)
target_link_libraries(io MPI::MPI_CXX)

# compile solution if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/io_solution.cpp")
  add_executable(io_solution io_solution.cpp)
  target_link_libraries(io_solution MPI::MPI_CXX)
endif()

# make pack_assignment target
add_custom_target(pack_assignment
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/io.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/lena_in.pgm"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# make pack_solution target
add_custom_target(pack_solution
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/io.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/io_solution.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/lena_in.pgm"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")