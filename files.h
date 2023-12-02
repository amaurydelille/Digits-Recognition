#pragma once
#include <err.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SIZE 784
#define DIRPATH "training"
#define INPUTS 10
#define IMAGES 10

SDL_Surface* load_image(const char* path);
Uint32 pixel_to_grayscale(Uint32 pixel_color, SDL_PixelFormat* format);
double* get_formated_image(SDL_Surface* surface);
double *path_to_input(char* path);
void printdigit(double* image);
