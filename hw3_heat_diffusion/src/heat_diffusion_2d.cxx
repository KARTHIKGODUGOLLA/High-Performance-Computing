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
        // initialize the arrays used to store the row from our above, below,left and right neighbors

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

    // Check command-line arguments length
    if (argc != 6) {
        cerr << "Invalid arguments." << endl;
        cerr << "Proper usage: " << argv[0] << " <simulation_name> <simulation_x> <simulation_y> <x_split> <y_split>" << endl;
        MPI_Finalize();
        return 1;
    }

    // Read command-line arguments
    string simulation_name = argv[1];
    simulation_name += " Process " + std::to_string(process_id);

    int32_t simulation_x = atoi(argv[2]);
    int32_t simulation_y = atoi(argv[3]);
    int32_t x_split = atoi(argv[4]);
    int32_t y_split = atoi(argv[5]);

    // displays error message if the number of processes created is not equal to y_split * x_split.
    if (number_processes != x_split * y_split)
    {
        cerr << "Error: Number of processes should be equal to x_split * y_split." << endl;
        MPI_Finalize();
        return 1;
    }

    // Calculate the slice dimensions and offsets
    int32_t slice_x = simulation_x / x_split;
    int32_t slice_y = simulation_y / y_split;
    int32_t x_offset = (process_id % x_split) * slice_x;
    int32_t y_offset = (process_id / x_split) * slice_y;

    // Initialize SDL
    SDL_Event event;
    SDL_Renderer *renderer;
    SDL_Window *window;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(slice_x, slice_y, 0, &window, &renderer);
    SDL_SetWindowTitle(window, simulation_name.c_str());
    SDL_SetWindowPosition(window, 100 + (slice_x + 40) * (process_id % x_split),
                          100 + (slice_y + 40) * (process_id / x_split));

    // Create a SimulationSlice instance
    SimulationSlice slice(process_id, number_processes, slice_y, slice_x, y_offset, x_offset, x_split, y_split, renderer);

    // Initialize heat blocks for each process based on their offset
for (int32_t y = y_offset; y < y_offset + slice_y; y++) {
    for (int32_t x = x_offset; x < x_offset + slice_x; x++) {
        //creating three blocks of heat at the top right, center and bottom
        if ((y < simulation_y * (1.0 / 5.0) && x >= simulation_x * (3.0 / 5.0)) || 
            (y >= simulation_y * (1.0 / 5.0) && y < simulation_y * (3.0 / 5.0) && x >= simulation_x / 4 && x < simulation_x / 2) || 
            (y >= simulation_y * (9.25 / 10.0) && x >= simulation_x * (4.0 / 10.0) && x < simulation_x * (8.0 / 10.0))) {
            slice.set_global_pixel(y, x, 1.0);
        }
    }
}
    int32_t iteration = 0;
    while (true) {

        slice.send_top();
        slice.send_bottom();
        slice.send_left();
        slice.send_right();

        slice.receive_top();
        slice.receive_bottom();
        slice.receive_left();
        slice.receive_right();
        slice.iterate();

        slice.display_texture(renderer);

        if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
            break;

        cout << "[process " << process_id << "/" << number_processes << "] iteration: " << iteration << endl;
        iteration++;
        if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
            break;
    }

    // Cleanup
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
