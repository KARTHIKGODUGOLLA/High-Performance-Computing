cmake_minimum_required (VERSION 3.8)
project (DSCI_650_HW1)

# The version number.
set (SOM_VERSION_MAJOR 0)
set (SOM_VERSION_MINOR 1)

#add_definitions( -DEXACT_VERSION="${EXACT_VERSION_MAJOR}.${EXACT_VERSION_MINOR}" )

set(PLATFORM 64)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# can try these out for debugging

#SET (CMAKE_CXX_FLAGS                "-std=gnu++17 -Wall -O3 -funroll-loops  -msse3 -fsanitize=address -fno-omit-frame-pointer -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=c++20 -Wall -O3 -funroll-loops  -msse3 -fsanitize=address -fno-omit-frame-pointer -D_GLIBCXX_DEBUG")

SET (CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS} -Wall -O4 -funroll-loops -DNDEBUG")

add_subdirectory(src)
add_subdirectory(test)

enable_testing()

add_test(NAME test_sample COMMAND bin/test_sample)

add_test(NAME test_database_statistics COMMAND bin/test_database_statistics)
add_test(NAME test_database_normalize COMMAND bin/test_database_normalize)

add_test(NAME test_distance_function COMMAND bin/test_distance_function)
add_test(NAME test_get_bmu COMMAND bin/test_get_bmu)
add_test(NAME test_neighborhood_function COMMAND bin/test_neighborhood_function)
add_test(NAME test_get_neighbors COMMAND bin/test_get_neighbors)
add_test(NAME test_update_cell COMMAND bin/test_update_cell)


