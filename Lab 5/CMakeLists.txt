#
# @file   CMakeLists.txt
#
# @author David Bayer \n
#         Faculty of Information Technology \n
#         Brno University of Technology \n
#         ibayer@fit.vutbr.cz
#
# @brief  CMakeLists for lab5
#
# @date   19 March 2024, 17:00 (created) \n
#

cmake_minimum_required(VERSION 3.20)

# project name
project(lab5 LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find and setup MPI package
set(MPI_CXX_SKIP_MPICXX ON)

find_package(MPI REQUIRED COMPONENTS CXX)

# compile one
add_executable(one ${CMAKE_CURRENT_SOURCE_DIR}/one.cpp)
target_link_libraries(one MPI::MPI_CXX)

# compile solution if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/one_solution.cpp")
  add_executable(one_solution one_solution.cpp)
  target_link_libraries(one_solution MPI::MPI_CXX)
endif()

# make pack_assignment target
add_custom_target(pack_assignment
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/one.cpp"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# make pack_solution target
add_custom_target(pack_solution
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/one.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/one_solution.cpp"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")


