#
# @file   CMakeLists.txt
#
# @author David Bayer \n
#         Faculty of Information Technology \n
#         Brno University of Technology \n
#         ibayer@fit.vutbr.cz
#
# @brief  CMakeLists for lab3
#
# @date   05 March 2024, 17:00 (created) \n
#

cmake_minimum_required(VERSION 3.20)

# project name
project(lab3 LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find and setup MPI package
set(MPI_CXX_SKIP_MPICXX ON)

find_package(MPI REQUIRED COMPONENTS CXX)

# compile comm
add_executable(comm ${CMAKE_CURRENT_SOURCE_DIR}/comm.cpp)
target_link_libraries(comm MPI::MPI_CXX)

# compile solution if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/comm_solution.cpp")
  add_executable(comm_solution comm_solution.cpp)
  target_link_libraries(comm_solution MPI::MPI_CXX)
endif()

# make pack_assignment target
add_custom_target(pack_assignment
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/comm.cpp"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# make pack_solution target
add_custom_target(pack_solution
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/comm.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/comm_solution.cpp"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

