cmake_minimum_required(VERSION 3.20.0)

project(test_cmake_Eigen)

set(Eigen3_DIR D:/Work/Project_file/Cpp/Dependency/eigen-3.4.0/build)

find_package(Eigen3 3.4 REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)