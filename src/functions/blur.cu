#include "functions.cuh"

int parseBlurArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int radius = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (imgIn == nullptr || pathOut == nullptr || radius == 0)
    {
        handle_error("Invalid arguments for blur function.\n");
    }

    //TODO: leggere le immagini in base alla loro estensione
    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;


    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        blurSerial(img, pathOut, width, height, radius);
        free(img);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        blurOmp(img, pathOut, width, height, radius);
        free(img);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        blurCuda(img, pathOut, width, height, radius);
        free(img);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for blur function.\n");
    }
    free(tpp);

    return 0;
}

int blurSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius)
{
    return 0;
}

int blurOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius)
{
    return 0;
}

int blurCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius)
{
    return 0;
}