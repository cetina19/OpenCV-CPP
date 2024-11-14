cmake_minimum_required(VERSION 3.30)
set( CMAKE_CXX_COMPILER "C:/msys64/ucrt64/bin/g++" )
set( CMAKE_C_COMPILER "C:/msys64/ucrt64/bin/gcc" )
project(Task1)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Task1 main.cpp ./helpers/functions.cpp)

target_link_libraries(Task1 ${OpenCV_LIBS})