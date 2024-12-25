add_library(som dataset.cxx sample.cxx som.cxx cell.cxx)
add_executable(main main.cxx)
target_link_libraries(main som)
