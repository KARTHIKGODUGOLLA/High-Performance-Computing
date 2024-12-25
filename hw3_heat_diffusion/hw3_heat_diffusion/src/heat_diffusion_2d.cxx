#include <SDL2/SDL.h>
#include <string>
#include <iostream>
#include "mpi.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

class SimulationSlice {
public:
    int32_t slice_id, slice_max;
    int32_t y_size, x_size, y_offset, x_offset;
    float **pixels_current, **pixels_next;
    int32_t x_split, y_split;
    float *top_neighbor_bottom_row, *bottom_neighbor_top_row, *left_neighbor_right_col, *right_neighbor_left_col;
    SDL_Texture *texture;
    uint32_t *texture_buffer;

    SimulationSlice(int32_t _slice_id, int32_t _slice_max, int32_t _y_size, int32_t _x_size, int32_t _y_offset, int32_t _x_offset, int32_t _x_split, int32_t _y_split, SDL_Renderer *renderer)
        : slice_id(_slice_id), slice_max(_slice_max), y_size(_y_size), x_size(_x_size), y_offset(_y_offset), x_offset(_x_offset), x_split(_x_split), y_split(_y_split) {
        
        pixels_current = new float*[y_size];
        pixels_next = new float*[y_size];
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
        float sum;
        int32_t count;
        for (int32_t y = 1; y < y_size - 1; y++) {
            for (int32_t x = 1; x < x_size - 1; x++) {
                sum = pixels_current[y - 1][x] + pixels_current[y + 1][x] +
                      pixels_current[y][x - 1] + pixels_current[y][x + 1];
                pixels_next[y][x] = sum / 4.0;
            }
        }

        // Handle boundary cells using the neighbor data
        for (int32_t x = 0; x < x_size; x++) {
            // Top boundary
            float top_sum = top_neighbor_bottom_row[x] + pixels_current[1][x];
            top_sum += (x > 0 ? pixels_current[0][x - 1] : 0) + (x < x_size - 1 ? pixels_current[0][x + 1] : 0);
            pixels_next[0][x] = top_sum / 4.0;

            // Bottom boundary
            float bottom_sum = bottom_neighbor_top_row[x] + pixels_current[y_size - 2][x];
            bottom_sum += (x > 0 ? pixels_current[y_size - 1][x - 1] : 0) + (x < x_size - 1 ? pixels_current[y_size - 1][x + 1] : 0);
            pixels_next[y_size - 1][x] = bottom_sum / 4.0;
        }

        for (int32_t y = 0; y < y_size; y++) {
            // Left boundary
            float left_sum = left_neighbor_right_col[y] + pixels_current[y][1];
            left_sum += (y > 0 ? pixels_current[y - 1][0] : 0) + (y < y_size - 1 ? pixels_current[y + 1][0] : 0);
            pixels_next[y][0] = left_sum / 4.0;

            // Right boundary
            float right_sum = right_neighbor_left_col[y] + pixels_current[y][x_size - 2];
            right_sum += (y > 0 ? pixels_current[y - 1][x_size - 1] : 0) + (y < y_size - 1 ? pixels_current[y + 1][x_size - 1] : 0);
            pixels_next[y][x_size - 1] = right_sum / 4.0;
        }

        // Swap current and next buffers
        float **temp = pixels_current;
        pixels_current = pixels_next;
        pixels_next = temp;
    }

    void send_top() {
        if (slice_id >= x_split) {
            MPI_Send(pixels_current[0], x_size, MPI_FLOAT, slice_id - x_split, 0, MPI_COMM_WORLD);
        }
    }

    void send_bottom() {
        if (slice_id < slice_max - x_split) {
            MPI_Send(pixels_current[y_size - 1], x_size, MPI_FLOAT, slice_id + x_split, 0, MPI_COMM_WORLD);
        }
    }

    void send_left() {
        if (slice_id % x_split != 0) {
            for (int32_t y = 0; y < y_size; y++) {
                MPI_Send(&pixels_current[y][0], 1, MPI_FLOAT, slice_id - 1, 0, MPI_COMM_WORLD);
            }
        }
    }

    void send_right() {
        if (slice_id % x_split != x_split - 1) {
            for (int32_t y = 0; y < y_size; y++) {
                MPI_Send(&pixels_current[y][x_size - 1], 1, MPI_FLOAT, slice_id + 1, 0, MPI_COMM_WORLD);
            }
        }
    }

    void receive_top() {
        if (slice_id >= x_split) {
            MPI_Recv(top_neighbor_bottom_row, x_size, MPI_FLOAT, slice_id - x_split, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    void receive_bottom() {
        if (slice_id < slice_max - x_split) {
            MPI_Recv(bottom_neighbor_top_row, x_size, MPI_FLOAT, slice_id + x_split, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    void receive_left() {
        if (slice_id % x_split != 0) {
            for (int32_t y = 0; y < y_size; y++) {
                MPI_Recv(&left_neighbor_right_col[y], 1, MPI_FLOAT, slice_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    void receive_right() {
        if (slice_id % x_split != x_split - 1) {
            for (int32_t y = 0; y < y_size; y++) {
                MPI_Recv(&right_neighbor_left_col[y], 1, MPI_FLOAT, slice_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    void display_texture(SDL_Renderer *renderer) {
        uint8_t blue = 0;
        uint8_t green = 0;
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
        cerr << "Usage: " << argv[0] << " <simulation_name> <simulation_x> <simulation_y> <x_split> <y_split>" << endl;
        exit(1);
    }

    string simulation_name = argv[1];
    int32_t simulation_x = atoi(argv[2]);
    int32_t simulation_y = atoi(argv[3]);
    int32_t x_split = atoi(argv[4]);
    int32_t y_split = atoi(argv[5]);

    if (number_processes != x_split * y_split) {
        cerr << "Number of processes must be equal to x_split * y_split" << endl;
        exit(1);
    }
    // Initialize SDL
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("Heat Diffusion", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    // Calculate the slice dimensions and offsets
    int32_t slice_x = simulation_x / x_split;
    int32_t slice_y = simulation_y / y_split;
    int32_t x_offset = (process_id % x_split) * slice_x;
    int32_t y_offset = (process_id / x_split) * slice_y;

    // Create SimulationSlice instance
    SimulationSlice slice(process_id, number_processes, slice_y, slice_x, y_offset, x_offset, renderer);

    // Simulation loop (example)
    for (int iteration = 0; iteration < /* Number of iterations */; ++iteration) {
        // Compute heat diffusion logic here

        // Send neighbors
        slice.send_neighbors(x_split, y_split);

        // Rendering logic
        SDL_RenderClear(renderer);
        // Render your simulation data here
        SDL_RenderPresent(renderer);
        
        // Optional: delay for visualization
        SDL_Delay(100); // Delay in milliseconds
    }

    // Cleanup
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    MPI_Finalize();
    return 0;
}
