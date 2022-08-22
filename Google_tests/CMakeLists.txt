project(Google_tests)
add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(TESTS basic_test.c)
add_executable(Google_Tests_run ${TESTS})

target_link_libraries(Google_Tests_run src)

target_link_libraries(Google_Tests_run gtest gtest_main)