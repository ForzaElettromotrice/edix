//
// Created by f3m on 19/01/24.
//

#include "functions.cuh"

int parseDownscaleArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int factor = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (imgIn == nullptr || pathOut == nullptr || factor == 0)
    {
        handle_error("Invalid arguments for downscale function.\n");
    }
    // TODO: leggere le immagini in base all'estenzione

    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;
    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        downscaleSerial(img, pathOut, width, height, factor);
        free(img);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        downscaleOmp(img, pathOut, width, height, factor);
        free(img);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        downscaleCuda(img, pathOut, width, height, factor);
        free(img);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for downscale function.\n");
    }

    free(tpp);
    return 0;
}

int downscaleSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}

int downscaleOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}

int downscaleCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}