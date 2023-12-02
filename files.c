#include <stdio.h>
#include <err.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <dirent.h>
#include <string.h>

#define INPUTS 10
#define IMAGES 10

SDL_Surface* load_image(const char* path) {
    SDL_Surface* temp = IMG_Load(path);
    SDL_Surface* surface = malloc(sizeof(SDL_Surface));
    surface = SDL_ConvertSurfaceFormat(temp, SDL_PIXELFORMAT_RGB888, 0);
    SDL_FreeSurface(temp);

    if (surface == NULL)
        errx(EXIT_FAILURE, "%s, image non existante : %s", SDL_GetError(), path);

    return surface;
}

double pixel_to_grayscale(Uint32 pixel_color, SDL_PixelFormat* format) {
    Uint8 r, g, b;
    SDL_GetRGB(pixel_color, format, &r, &g, &b);
    return (0.3 * r + 0.59 * g + 0.11 * b) / 255.0;
}


double* get_formated_image(SDL_Surface* surface) {
    int len = surface->w * surface->h;
    SDL_PixelFormat* format = surface->format;
    Uint32* pixels = (Uint32*)surface->pixels;

    double* output = malloc(sizeof(double) * len);
    if (output == NULL) {
        errx(EXIT_FAILURE, "Failed to allocate memory for output");
    }

    int lock = SDL_LockSurface(surface);
    if (lock) {
        free(output);
        errx(EXIT_FAILURE, "%s", SDL_GetError());
    }

    for (int i = 0; i < len; i++) {
        output[i] = pixel_to_grayscale(pixels[i], format);
    }

    SDL_UnlockSurface(surface);

    return output;
}

double* path_to_input(char* path) {
    SDL_Surface* surface = load_image(path);
    double* inputs = get_formated_image(surface);
    return inputs;
}

void printdigit(double* image) {
    for(size_t i = 0; i < 28; i++){
        for(size_t j = 0; j < 28; j++){
            double pixel = image[i * 28 + j]; 
            if (pixel == 0)
                printf("  ");
            else if (pixel < 0.5)
                printf(". ");
            else
                printf("X ");
        }
        printf("\n");
    }
}