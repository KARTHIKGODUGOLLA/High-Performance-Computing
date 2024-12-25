add_executable(heat_diffusion heat_diffusion.cxx)
target_link_libraries(heat_diffusion ${SDL2_LIBRARIES})

add_executable(heat_diffusion_mpi heat_diffusion_mpi.cxx)
target_link_libraries(heat_diffusion_mpi ${MPI_LIBRARIES} ${MPI_EXTRA} ${SDL2_LIBRARIES})

add_executable(heat_diffusion_2d heat_diffusion_2d.cxx)
target_link_libraries(heat_diffusion_2d ${MPI_LIBRARIES} ${MPI_EXTRA} ${SDL2_LIBRARIES})

add_executable(mpi_practice mpi_practice.cxx)
target_link_libraries(mpi_practice ${MPI_LIBRARIES} ${MPI_EXTRA} ${SDL2_LIBRARIES})

