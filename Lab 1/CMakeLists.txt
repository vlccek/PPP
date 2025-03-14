#
# @file   CMakeLists.txt
#
# @author David Bayer \n
#         Faculty of Information Technology \n
#         Brno University of Technology \n
#         ibayer@fit.vutbr.cz
#
# @brief  CMakeLists for lab1
#
# @date   14 February 2024, 14:00 (created) \n
#

cmake_minimum_required(VERSION 3.20)

# project name
project(lab1 LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find and setup MPI package
set(MPI_CXX_SKIP_MPICXX ON)

find_package(MPI REQUIRED COMPONENTS CXX)

# check if solution exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/p2p_solution.cpp)
  set(HAS_SOLUTION TRUE)
else()
  set(HAS_SOLUTION FALSE)
endif()

# compile p2p
add_executable(p2p ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cpp)
target_link_libraries(p2p MPI::MPI_CXX)

# compile solution if it exists
if(HAS_SOLUTION)
  add_executable(p2p_solution ${CMAKE_CURRENT_SOURCE_DIR}/p2p_solution.cpp)
  target_link_libraries(p2p_solution MPI::MPI_CXX)
endif()

# make pack_assignment target
add_custom_target(pack_assignment
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/p2p.cpp")

# make pack_solution target
if(HAS_SOLUTION)
  add_custom_target(pack_solution
    COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}-solution.zip" --format=zip
                                         "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                         "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                         "${CMAKE_CURRENT_SOURCE_DIR}/p2p.cpp"
                                         "${CMAKE_CURRENT_SOURCE_DIR}/p2p_solution.cpp")
endif()
