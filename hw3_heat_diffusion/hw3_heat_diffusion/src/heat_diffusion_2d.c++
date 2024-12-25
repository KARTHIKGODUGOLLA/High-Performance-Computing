#include <SDL2/SDL.h>

#include <string>
using std::string;
using std::to_string;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "mpi.h"

class SimulationSlice {
public:
    int32_t slice_id;
    int32_t slice_max;
    int32_t y_size;
    int32_t x_size;
    int32_t y_offset;
    int32_t x_offset;

    float **pixels_current;
    float **pixels_next;

    float *top_neighbor_bottom_row;
    float *bottom_neighbor_top_row;
    float *left_neighbor_right_col;
    float *right_neighbor_left_col;

    SDL_Texture *texture;
    uint32_t *texture_buffer;

    SimulationSlice(int32_t _slice_id, int32_t _slice_max, int32_t _y_size, int32_t _x_size, int32_t _y_offset, int32_t _x_offset, SDL_Renderer *renderer)
        : slice_id(_slice_id), slice_max(_slice_max), y_size(_y_size), x_size(_x_size), y_offset(_y_offset), x_offset(_x_offset) {

        pixels_current = new float *[y_size];
        pixels_next = new float *[y_size];
        for (int32_t y = 0; y < y_size; y++) {
            pixels_current[y] = new float[x_size];
            pixels_next[y] = new float[x_size];
            for (int32_t x = 0; x < x_size; x++) {
                pixels_current[y][x] = 0;
                pixels_next[y][x] = 0;
            }
        }

        top_neighbor_bottom_row = new float[x_size];
        bottom_neighbor_top_row = new float[x_size];
        left_neighbor_right_col = new float[y_size];
        right_neighbor_left_col = new float[y_size];

        for (int32_t x = 0; x < x_size; x++) {
            top_neighbor_bottom_row[x] = 0;
            bottom_neighbor_top_row[x] = 0;
        }
        for (int32_t y = 0; y < y_size; y++) {
            left_neighbor_right_col[y] = 0;
            right_neighbor_left_col[y] = 0;
        }

        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, x_size, y_size);
        texture_buffer = new uint32_t[x_size * y_size];
    }

    void set_global_pixel(int32_t y, int32_t x, float value) {
        if (y >= y_offset && y < y_offset + y_size && x >= x_offset && x < x_offset + x_size) {
            pixels_current[y - y_offset][x - x_offset] = value;
        }
    }

    void iterate() {
        for (int32_t y = 1; y < y_size - 1; y++) {
            for (int32_t x = 1; x < x_size - 1; x++) {
                pixels_next[y][x] = 0.25f * (pixels_current[y - 1][x] + pixels_current[y + 1][x] + pixels_current[y][x - 1] + pixels_current[y][x + 1]);
            }
        }
        std::swap(pixels_current, pixels_next);
    }

    void send_neighbors(int32_t x_split, int32_t y_split) {
        // Implement MPI_Send for top, bottom, left, right neighbors
        if (y_offset > 0) MPI_Send(pixels_current[0], x_size, MPI_FLOAT, slice_id - x_split, 0, MPI_COMM_WORLD); // Send top row
        if (y_offset + y_size < y_split) MPI_Send(pixels_current[y_size - 1], x_size, MPI_FLOAT, slice_id + x_split, 0, MPI_COMM_WORLD); // Send bottom row
        if (x_offset > 0) MPI_Send(left_neighbor_right_col, y_size, MPI_FLOAT, slice_id - 1, 0, MPI_COMM_WORLD); // Send left column
        if (x_offset + x_size < x_split) MPI_Send(right_neighbor_left_col, y_size, MPI_FLOAT, slice_id + 1, 0, MPI_COMM_WORLD); // Send right column
    }

    void receive_neighbors(int32_t x_split, int32_t y_split) {
        // Implement MPI_Recv for top, bottom, left, right neighbors
        if (y_offset > 0) MPI_Recv(top_neighbor_bottom_row, x_size, MPI_FLOAT, slice_id - x_split, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive top row
        if (y_offset + y_size < y_split) MPI_Recv(bottom_neighbor_top_row, x_size, MPI_FLOAT, slice_id + x_split, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive bottom row
        if (x_offset > 0) MPI_Recv(left_neighbor_right_col, y_size, MPI_FLOAT, slice_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive left column
        if (x_offset + x_size < x_split) MPI_Recv(right_neighbor_left_col, y_size, MPI_FLOAT, slice_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive right column
    }

    void display_texture(SDL_Renderer *renderer) {
        uint8_t blue = 0, green = 0;
        for (int32_t y = 0; y < y_size; y++) {
            for (int32_t x = 0; x < x_size; x++) {
                uint8_t red = 255 * pixels_current[y][x];
                texture_buffer[(y * x_size) + x] = 0xFF000000 | (red << 16) | (blue << 8) | green;
            }
        }
        SDL_UpdateTexture(texture, NULL, texture_buffer, x_size * sizeof(uint32_t));
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }
};

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int32_t number_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &number_processes);

    int32_t process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    if (argc != 6) {
        cerr << "Invalid arguments." << endl;
        cerr << "Proper usage: " << argv[0] << " <simulation_name> <simulation_x> <simulation_y> <x_split> <y_split>" << endl;
        MPI_Finalize();
        return 1;
    }

    string simulation_name = argv[1];
    simulation_name += " Process " + to_string(process_id);

    int32_t simulation_x = atoi(argv[2]);
    int32_t simulation_y = atoi(argv[3]);
    int32_t x_split = atoi(argv[4]);
    int32_t y_split = atoi(argv[5]);

    if (number_processes != x_split * y_split) {
        cerr << "Error: Number of processes must equal x_split * y_split." << endl;
        MPI_Finalize();
        return 1;
    }

    int32_t slice_x = simulation_x / x_split;
    int32_t slice_y = simulation_y / y_split;
    int32_t x_offset = (process_id % x_split) * slice_x;
    int32_t y_offset = (process_id / x_split) * slice_y;

    SDL_Event event;
    SDL_Renderer *renderer;
    SDL_Window *window;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(slice_x, slice_y, 0, &window, &renderer);
    SDL_SetWindowTitle(window, simulation_name.c_str());
    SDL_SetWindowPosition(window, 100 + (slice_x + 40) * (process_id % x_split), 100 + (slice_y + 40) * (process_id / x_split));

    SimulationSlice slice(process_id, number_processes, slice_y, slice_x, y_offset, x_offset, renderer);

    for (int y = 0; y < simulation_y * (1.0 / 5.0); y++) {
        for (int x = simulation_x * (3.0 / 5.0); x < simulation_x; x++) {
            slice.set_global_pixel(y, x, 1.0);
        }
    }

    int32_t iteration = 0;
    while (1) {
        if (iteration % 100 == 0) slice.display_texture(renderer);
        slice.send_neighbors(x_split, y_split);
        slice.receive_neighbors(x_split, y_split);
        slice.iterate();

        while (SDL_PollEvent(&event)) if (event.type == SDL_QUIT) break;
        iteration++;
    }

    MPI_Finalize();
    return 0;
}
