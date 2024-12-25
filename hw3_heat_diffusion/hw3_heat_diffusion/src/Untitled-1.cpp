#include <SDL2/SDL.h>
#include <string>
using std::string;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
using std::cin;

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Invalid arguments." << endl;
        cerr << "Proper usage:" << endl;
        cerr << "\t" << argv[0] << " <window_name> <window_width> <window_height>" << endl;
        exit(1);
    }

    string window_name = argv[1];
    int32_t window_width = atoi(argv[2]);
    int32_t window_height = atoi(argv[3]);

    int32_t window_size = window_width * window_height;

    // Initialize the simulation heat values
    float **pixels_current = new float*[window_height];
    float **pixels_next = new float*[window_height];

    for (int32_t y = 0; y < window_height; y++) {
        pixels_current[y] = new float[window_width];
        pixels_next[y] = new float[window_width];

        for (int32_t x = 0; x < window_width; x++) {
            pixels_current[y][x] = 0;
            pixels_next[y][x] = 0;
        }
    }

    // Create a block of heat in top right corner
    for (int y = 0; y < window_height * (1.0/5.0); y++) {
        for (int x = window_width * (3.0/4.0); x < window_width; x++) {
            pixels_current[y][x] = 1.0;
        }
    }

    // Create a block of heat in the center
    for (int y = window_height / 2; y < window_height * (3.0/4.0); y++) {
        for (int x = window_width / 4; x < window_width / 2; x++) {
            pixels_current[y][x] = 1.0;
        }
    }

    SDL_Event event;
    SDL_Renderer *renderer;
    SDL_Window *window;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(window_width, window_height, 0, &window, &renderer);
    SDL_SetWindowTitle(window, window_name.c_str());

    SDL_Texture *heat_display = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, window_width, window_height);
    uint32_t *heat_display_buffer = new uint32_t[window_size];

    int32_t iteration = 0;
    while (1) {
        // this sets the texture we'll use to display the simulation state
        uint8_t blue = 0;
        uint8_t green = 0;
        for (int32_t y = 0; y < window_height; y++) {
            for (int32_t x = 0; x < window_width; x++) {
                uint8_t red = 255 * pixels_current[y][x];
                heat_display_buffer[(y * window_width) + x] = 0xFF000000 | (red << 16) | (blue << 8) | green;

                //SDL_SetRenderDrawColor(renderer, 255 * pixels_current[y][x] /*red*/, 0 /*green*/, 0 /*blue*/, 255 /*opacity*/);
                //SDL_RenderDrawPoint(renderer, x, y);
            }
        }
        SDL_UpdateTexture(heat_display, NULL, heat_display_buffer, window_width * sizeof(uint32_t));

        //Render texture to screen
        SDL_RenderCopy(renderer, heat_display, NULL, NULL );

        SDL_RenderPresent(renderer);

        // perform one iteration of updating the simulations heat values. the next values
        // will be set to the average of the up/down/left/right cells around it.
    float sum = 0.0;
    int32_t count = 0;
    int max_iterations = 1000;
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        for (int32_t y = 0; y < window_height; y++) {
            for (int32_t x = 0; x < window_width; x++) {
                sum = 0.0;
                count = 0;

                if (y > 0) {
                    sum += pixels_current[y-1][x];
                    count += 1;
                }
                if (y < window_height - 1) {
                    sum += pixels_current[y+1][x];
                    count += 1;
                }
                if (x > 0) {
                    sum += pixels_current[y][x-1];
                    count += 1;
                }
                if (x < window_width - 1) {
                    sum += pixels_current[y][x+1];
                    count += 1;
                }

                pixels_next[y][x] = sum / count;
            }
        }

        // Swap pixel buffers
        float **temp = pixels_current;
        pixels_current = pixels_next;
        pixels_next = temp;
    }

    // Wait for the user to press Enter before exiting
    cout << "Press Enter to exit..." << endl;
    std::cin.get();

    // Clean up
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
