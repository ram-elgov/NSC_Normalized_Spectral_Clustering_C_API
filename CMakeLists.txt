cmake_minimum_required(VERSION 3.22)
project(NSC___Normalized_Spectral_Clustering_C_API C)
set(CMAKE_C_STANDARD 90)
add_executable(NSC___Normalized_Spectral_Clustering_C_API src/spkmeans.c src/spkmeans.h src/spkmeansmodule_error.c src/spkmeansmodule.c)
target_link_libraries(NSC___Normalized_Spectral_Clustering_C_API m)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES src/spkmeans.c Google_tests/basic_test.c)
add_executable(NSC_run ${SOURCE_FILES})
include_directories(src)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_subdirectory(src)

target_link_libraries(NSC_run src)

add_subdirectory(Google_tests)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(NSC_run ${PYTHON_LIBRARIES})