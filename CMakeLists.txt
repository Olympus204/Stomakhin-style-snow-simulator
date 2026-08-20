cmake_minimum_required(VERSION 3.20)

project(snow_simulator LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_library(snow_core
    src/grid.cpp
    src/collision.cpp
    src/import.cpp
    src/export.cpp
    src/debug.cpp
)

target_include_directories(snow_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(snow_core PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

target_compile_options(snow_core PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(snow_sim src/main.cpp)
target_link_libraries(snow_sim PRIVATE snow_core)

add_executable(snow_tests tests/snow_tests.cpp)
target_link_libraries(snow_tests PRIVATE snow_core)

enable_testing()
add_test(NAME snow_tests COMMAND snow_tests)
