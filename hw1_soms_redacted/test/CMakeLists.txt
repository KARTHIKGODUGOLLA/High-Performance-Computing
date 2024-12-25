include_directories(${DSCI_650_HW1_SOURCE_DIR}/src)

add_executable(test_sample test_sample.cxx test_common.cxx)
target_link_libraries(test_sample som)

add_executable(test_database_statistics test_database_statistics.cxx test_common.cxx)
target_link_libraries(test_database_statistics som)

add_executable(test_database_normalize test_database_normalize.cxx test_common.cxx)
target_link_libraries(test_database_normalize som)

add_executable(test_distance_function test_distance_function.cxx test_common.cxx)
target_link_libraries(test_distance_function som)

add_executable(test_get_bmu test_get_bmu.cxx test_common.cxx)
target_link_libraries(test_get_bmu som)

add_executable(test_neighborhood_function test_neighborhood_function.cxx test_common.cxx)
target_link_libraries(test_neighborhood_function som)

add_executable(test_get_neighbors test_get_neighbors.cxx test_common.cxx)
target_link_libraries(test_get_neighbors som)

add_executable(test_update_cell test_update_cell.cxx test_common.cxx)
target_link_libraries(test_update_cell som)


