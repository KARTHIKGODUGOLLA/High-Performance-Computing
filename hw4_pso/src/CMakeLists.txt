add_subdirectory(objectives)

add_executable(particle_swarm_optimization particle_swarm_optimization.cxx)
add_executable(mpi_asynch_mw mpi_asynch_mw.cxx)


target_link_libraries(particle_swarm_optimization objectives ${MPI_LIBRARIES} ${MPI_EXTRA})
target_link_libraries(mpi_asynch_mw objectives ${MPI_LIBRARIES} ${MPI_EXTRA})

