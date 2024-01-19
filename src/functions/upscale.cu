//
// Created by f3m on 19/01/24.
//

#include "functions.cuh"

int parseUpscaleArgs(char *args)
{
    char *img1 = strtok(args, ",");
    char *pathOut = strtok(nullptr, ",");
    int factor = (int) strtol(strtok(nullptr, ","), nullptr, 10);

    if (img1 == nullptr || pathOut == nullptr || factor == 0)
    {
        handle_error("Invalid arguments for upscale\n");
    }

    // TODO: leggere le immagini in base all'estenzione
    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;

    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img = loadPPM(img1, &width, &height);
        upscaleSerial(img, pathOut, width, height, factor);
        free(img);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img = loadPPM(img1, &width, &height);
        upscaleOmp(img, pathOut, width, height, factor);
        free(img);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img = loadPPM(img1, &width, &height);
        upscaleCuda(img, pathOut, width, height, factor);
        free(img);
    } else
    {
        free(tpp);
        handle_error("Invalid TPP\n");
    }
    free(tpp);
    return 0;
}

int upscaleSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}

int upscaleOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}

int upscaleCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}