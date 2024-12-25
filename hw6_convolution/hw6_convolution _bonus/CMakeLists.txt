cmake_minimum_required (VERSION 3.8)
project (DSCI_650_HW1 LANGUAGES CXX CUDA)

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

add_test(NAME test_tensor COMMAND bin/test_tensor)
add_test(NAME test_convolution_2d COMMAND bin/test_convolution_2d)
add_test(NAME test_convolution_2d_large COMMAND bin/test_convolution_2d_large)
