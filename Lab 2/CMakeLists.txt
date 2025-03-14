#
# @file   CMakeLists.txt
#
# @author David Bayer \n
#         Faculty of Information Technology \n
#         Brno University of Technology \n
#         ibayer@fit.vutbr.cz
#
# @brief  CMakeLists for lab2
#
# @date   14 February 2024, 14:00 (created) \n
#

cmake_minimum_required(VERSION 3.20)

# project name
project(lab2 LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find and setup MPI package
set(MPI_CXX_SKIP_MPICXX ON)

find_package(MPI REQUIRED COMPONENTS CXX)

# compile bcast
add_executable(bcast ${CMAKE_CURRENT_SOURCE_DIR}/bcast.cpp)
target_link_libraries(bcast MPI::MPI_CXX)

# compile scatter
add_executable(scatter ${CMAKE_CURRENT_SOURCE_DIR}/scatter.cpp)
target_link_libraries(scatter MPI::MPI_CXX)

# compile solution if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/bcast_solution.cpp")
  add_executable(bcast_solution bcast_solution.cpp)
  target_link_libraries(bcast_solution MPI::MPI_CXX)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/scatter_solution.cpp")
  add_executable(scatter_solution scatter_solution.cpp)
  target_link_libraries(scatter_solution MPI::MPI_CXX)
endif()

# make pack_assignment target
add_custom_target(pack_assignment
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/bcast.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/scatter.cpp")

# make pack_solution target
add_custom_target(pack_solution
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/PPP-${PROJECT_NAME}.zip" --format=zip
                                       "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/doc.pdf"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/bcast.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/scatter.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/bcast_solution.cpp"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/scatter_solution.cpp")
