#include <SDL2/SDL.h>

#include <string>
using std::string;
using std::to_string;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "mpi.h"

class SimulationSlice
{
public:
    // the slice id of this slice (also the process id)
    int32_t slice_id;

    // the total number of slices in the simulation
    int32_t slice_max;

    // the number of pixels/cells in the y dimension
    int32_t y_size;

    // the number of pixels/cells in the y dimension
    int32_t x_size;

    // this slices offset from the 0th y index
    int32_t y_offset;

    // two sets of arrays for the pixels, one for the current values
    // and another to set to the next values (which then get swapped)
    float **pixels_current;
    float **pixels_next;

    float *top_neighbor_bottom_row;
    float *bottom_neighbor_top_row;

    // texture map used to display to the GUI
    SDL_Texture *texture;

    // buffer for the actual texture pixel values
    uint32_t *texture_buffer;

    /**
     * Constructs a horizontal slice of our heat simulation.
     *
     * \param _y_size is the height of the slice of the simulation
     * \param _x_size is the width of the slice of the simulation
     * \param y_offset is how many pixels (cells) offset we are from the
     *      top of the simulation
     * \param renderer is the SDL_Renderer used to display this slice of
     *      the simulation
     */
    SimulationSlice(int32_t _slice_id, int32_t _slice_max, int32_t _y_size, int32_t _x_size, int32_t _y_offset, SDL_Renderer *renderer) : slice_id(_slice_id), slice_max(_slice_max), y_size(_y_size), x_size(_x_size), y_offset(_y_offset)
    {

        // initialize the other arrays and set default values
        pixels_current = new float *[y_size];
        pixels_next = new float *[y_size];
        for (int32_t y = 0; y < y_size; y++)
        {

            pixels_current[y] = new float[x_size];
            pixels_next[y] = new float[x_size];

            for (int32_t x = 0; x < x_size; x++)
            {
                pixels_current[y][x] = 0;
                pixels_next[y][x] = 0;
            }
        }

        // initialize the arrays used to store the row from our above and below neighbors
        // from MPI_Recv
        top_neighbor_bottom_row = new float[x_size];
        bottom_neighbor_top_row = new float[x_size];
        for (int32_t x = 0; x < x_size; x++)
        {
            top_neighbor_bottom_row[x] = 0;
            bottom_neighbor_top_row[x] = 0;
        }

        // initialize the texture for displaying this simulation slice
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, x_size, y_size);
        texture_buffer = new uint32_t[x_size * y_size];
    }

    /**
     *  Used in initial initialization of the simulation. Takes a global x/y
     *  value and sets that pixel (cell) to the given value if our slice contains
     *  that pixel/cell.
     *
     *  \param y is the global y index of the cell
     *  \param x is the global x index of the cell
     *  \param value is the heat value to set that cell to (if it is in this
     *      slice).
     */
    void set_global_pixel(int32_t y, int32_t x, float value)
    {
        if (y >= y_offset && y < y_offset + y_size)
        {
            pixels_current[y - y_offset][x] = value;
        }
    }

    /**
     * Performs one iteration of updating the simulation, calculating the next steps
     * heat values from the current heat values.
     */
    void iterate()
    {
        // TODO: Implement one iteration of the heat simulation
        float sum = 0.0;
        int32_t count = 0;
        for (int32_t y = 1; y < y_size - 1; y++)
        { // Avoid boundary cells (y = 0, y = y_size - 1)
            for (int32_t x = 1; x < x_size - 1; x++)
            { // Avoid boundary cells (x = 0, x = x_size - 1)
                sum = 0.0;
                count = 0;

                // Top neighbor
                if (y > 0)
                {
                    sum += pixels_current[y - 1][x];
                    count++;
                }

                // Bottom neighbor
                if (y < y_size - 1)
                {
                    sum += pixels_current[y + 1][x];
                    count++;
                }

                // Left neighbor
                if (x > 0)
                {
                    sum += pixels_current[y][x - 1];
                    count++;
                }

                // Right neighbor
                if (x < x_size - 1)
                {
                    sum += pixels_current[y][x + 1];
                    count++;
                }

                // Update the next state based on the average of the neighbors
                pixels_next[y][x] = sum / count;
            }
        }

        // Handle the boundary cells that depend on neighbors (top and bottom rows)
        for (int32_t x = 0; x < x_size; x++)
        {
            // Top row
            float top_sum = 0.0f;
            int count = 0;

            // Top neighbor from the adjacent process
            top_sum += top_neighbor_bottom_row[x];
            count++;

            // Bottom neighbor (next row in current process)
            top_sum += pixels_current[1][x];
            count++;

            // Left neighbor
            if (x > 0)
            {
                top_sum += pixels_current[0][x - 1];
                count++;
            }

            // Right neighbor
            if (x < x_size - 1)
            {
                top_sum += pixels_current[0][x + 1];
                count++;
            }

            // Average
            pixels_next[0][x] = top_sum / count;

            // Bottom row
            float bottom_sum = 0.0f;
            count = 0;

            // Top neighbor (previous row in current process)
            bottom_sum += pixels_current[y_size - 2][x];
            count++;

            // Bottom neighbor from the adjacent process
            bottom_sum += bottom_neighbor_top_row[x];
            count++;

            // Left neighbor
            if (x > 0)
            {
                bottom_sum += pixels_current[y_size - 1][x - 1];
                count++;
            }

            // Right neighbor
            if (x < x_size - 1)
            {
                bottom_sum += pixels_current[y_size - 1][x + 1];
                count++;
            }

            // Average
            pixels_next[y_size - 1][x] = bottom_sum / count;
        }

        //swap our two pixel buffers
        float **temp = pixels_current;
        pixels_current = pixels_next;
        pixels_next = temp;
    }

    /**
     * If we have a slice above us in the simulation, send our top row to it
     */
    void send_top()
    {
        // TODO: Implement sending our top row to the slice above us.
        if (slice_id > 0)
        {
            MPI_Send(pixels_current[0], x_size, MPI_FLOAT, slice_id - 1, 0, MPI_COMM_WORLD);
        }
    }

    /**
     * If we have a slice below us in the simulation, send our bottom row to it
     */
    void send_bottom()
    {
        // TODO: Implement sending our bottom row to the slice below us.
        if (slice_id < slice_max - 1)
        {
            MPI_Send(pixels_current[y_size - 1], x_size, MPI_FLOAT, slice_id + 1, 0, MPI_COMM_WORLD);
        }
    }

    /**
     * If we have a slice above us in the simulation, receive its bottom row
     */
    void receive_top()
    {
        // TODO: Implement receiving the bottom row from the slice above us.
        if (slice_id > 0)
        {
            MPI_Recv(top_neighbor_bottom_row, x_size, MPI_FLOAT, slice_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    /**
     * If we have a slice below us in the simulation, receive its top row
     */
    void receive_bottom()
    {
        // TODO: Implement receiving the top row from the slice below us.
        if (slice_id < slice_max - 1)
        { 
            MPI_Recv(bottom_neighbor_top_row, x_size, MPI_FLOAT, slice_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    /**
     * Updates the texture used to display this slice of the simulation.
     *
     * \param renderer is the SDL_Renderer used to display this slice of
     *      the simulation
     */
    void display_texture(SDL_Renderer *renderer)
    {
        uint8_t blue = 0;
        uint8_t green = 0;
        for (int32_t y = 0; y < y_size; y++)
        {
            for (int32_t x = 0; x < x_size; x++)
            {
                uint8_t red = 255 * pixels_current[y][x];
                texture_buffer[(y * x_size) + x] = 0xFF000000 | (red << 16) | (blue << 8) | green;

                // SDL_SetRenderDrawColor(renderer, 255 * pixels_current[y][x] /*red*/, 0 /*green*/, 0 /*blue*/, 255 /*opacity*/);
                // SDL_RenderDrawPoint(renderer, x, y);
            }
        }
        SDL_UpdateTexture(texture, NULL, texture_buffer, x_size * sizeof(uint32_t));

        // Render texture to screen
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int32_t number_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &number_processes);

    int32_t process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    if (argc != 4)
    {
        cerr << "Invalid arguments." << endl;
        cerr << "Proper usage:" << endl;
        cerr << "\t" << argv[0] << " <simulation_name> <simulation_x> <simulation_y>" << endl;
        exit(1);
    }

    string simulation_name = argv[1];
    simulation_name += " Process " + to_string(process_id);

    int32_t simulation_x = atoi(argv[2]);
    int32_t simulation_y = atoi(argv[3]);
    if (process_id == 0)
    {
        cout << "[process " << process_id << "/" << number_processes << "] simulation size y: " << simulation_y << ", x: " << simulation_x << endl;
    }

    int32_t slice_x = simulation_x;

    // TODO: Update the calculateion of slice_y and slice_y_offset such that it
    // correctly calculates them if the height of the simulation is not evenly
    // divisible by the number of processes.

    // slice the simulation in height by the number of processes
    int32_t slice_y = simulation_y / number_processes;
    int32_t slice_y_offset = process_id * slice_y;

    SDL_Event event;
    SDL_Renderer *renderer;
    SDL_Window *window;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(slice_x, slice_y, 0, &window, &renderer);
    SDL_SetWindowTitle(window, simulation_name.c_str());
    // set the windows with some padding between each of them
    // in the y dimension
    SDL_SetWindowPosition(window, 100, 100 + ((40 + slice_y) * process_id));

    SimulationSlice slice(process_id, number_processes, slice_y, slice_x, slice_y_offset, renderer);

    cout << "[process " << process_id << "/" << number_processes << "] y min: " << slice.y_offset << ", y_max : " << (slice.y_offset + slice.y_size) << endl;

    // create a block of heat in top right corner
    for (int y = 0; y < simulation_y * (1.0 / 5.0); y++)
    {
        for (int x = simulation_x * (3.0 / 5.0); x < simulation_x; x++)
        {
            slice.set_global_pixel(y, x, 1.0);
        }
    }

    // create a block of heat in the center
    for (int y = simulation_y * (1.0 / 5.0); y < simulation_y * (3.0 / 5.0); y++)
    {
        for (int x = simulation_x / 4; x < simulation_x / 2; x++)
        {
            slice.set_global_pixel(y, x, 1.0);
        }
    }

    // create a block of heat in the center
    for (int y = simulation_y * (9.25 / 10.0); y < simulation_y; y++)
    {
        for (int x = simulation_x * (4.0 / 10.0); x < simulation_x * (8.0 / 10.0); x++)
        {
            slice.set_global_pixel(y, x, 1.0);
        }
    }

    // loop for the simulation
    int32_t iteration = 0;
    while (1)
    {
        // transfer the initial top and bottom rows between slices
        slice.send_top();
        slice.send_bottom();
        slice.receive_top();
        slice.receive_bottom();

        // update the simulation
        slice.iterate();

        // display the updated simulation state
        slice.display_texture(renderer);

        cout << "[process " << process_id << "/" << number_processes << "] iteration: " << iteration << endl;
        iteration++;

        if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
            break;
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
