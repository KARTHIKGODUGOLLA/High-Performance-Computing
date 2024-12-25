cmake_minimum_required (VERSION 2.6)
project (DISPLAY_IMAGE LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_executable(nbody_gpu nbody_gpu.cxx)
target_link_libraries(nbody_gpu ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
add_executable(nbody_gpu_cuda nbody_gpu_cuda.cu)
target_link_libraries(nbody_gpu_cuda ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
