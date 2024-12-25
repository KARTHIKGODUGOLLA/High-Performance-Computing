som.cxx has an additional method hex_neighborhood_function to compute the neighborhood influence in hexagonal grids also added get_hex_neighbors method to calculate the neighbors in a hexagonal grid.

The train function in som.cxx is updated to accept a use_hex_grid flag as a parameter from input and will differentiate between the rectangular grid or hexagonal grid.