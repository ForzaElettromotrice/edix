//
// Created by f3m on 19/01/24.
//

#include "functions.cuh"

int parseColorFilterArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    //TODO: controllare che i valori siano compresi tra 0 e 255 e che non siano state inserite stringhe
    uint r = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);
    uint g = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);
    uint b = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);

    if (imgIn == nullptr || pathOut == nullptr || r > 255 || g > 255 || b > 255)
    {
        handle_error("Invalid arguments for color filter function.\n");
    }

    //TODO: leggere le immagini in base alla loro estensione
    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;

    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        colorFilterSerial(img, pathOut, width, height, r, g, b);
        free(img);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        colorFilterOmp(img, pathOut, width, height, r, g, b);
        free(img);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        colorFilterCuda(img, pathOut, width, height, r, g, b);
        free(img);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for color filter function.\n");
    }

    free(tpp);

    return 0;
}

int colorFilterSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, uint r, uint g, uint b)
{
    return 0;
}
int colorFilterOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, uint r, uint g, uint b)
{
    return 0;
}
int colorFilterCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, uint r, uint g, uint b)
{
    return 0;
}