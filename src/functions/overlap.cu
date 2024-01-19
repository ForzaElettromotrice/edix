//
// Created by f3m on 19/01/24.
//

#include "functions.cuh"

int parseOverlapArgs(char *args)
{
    char *img1 = strtok(args, ",");
    char *img2 = strtok(nullptr, ",");
    char *pathOut = strtok(nullptr, ",");
    //TODO: controllare se x e y sono stringhe o meno
    uint x = (uint) strtoul(strtok(nullptr, ","), nullptr, 10);
    uint y = (uint) strtoul(strtok(nullptr, ","), nullptr, 10);

    if (img1 == nullptr || img2 == nullptr || pathOut == nullptr)
    {
        handle_error("Invalid arguments for overlap\n");
    }
    // TODO: leggere le immagini in base all'estenzione

    char *tpp = getStrFromKey((char *) "TPP");
    uint width1;
    uint height1;
    uint width2;
    uint height2;

    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img1_1 = loadPPM(img1, &width1, &height1);
        unsigned char *img2_1 = loadPPM(img2, &width2, &height2);
        overlapSerial((img1_1), img2_1, pathOut, width1, height1, width2, height2, x, y);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img1_1 = loadPPM(img1, &width1, &height1);
        unsigned char *img2_1 = loadPPM(img2, &width2, &height2);
        overlapOmp(img1_1, img2_1, pathOut, width1, height1, width2, height2, x, y);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img1_1 = loadPPM(img1, &width1, &height1);
        unsigned char *img2_1 = loadPPM(img2, &width2, &height2);
        overlapCuda(img1_1, img2_1, pathOut, width1, height1, width2, height2, x, y);
    } else
    {
        free(tpp);
        handle_error("Invalid TPP\n");
    }
    free(tpp);
    return 0;
}

int overlapSerial(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2,
                  uint height2, uint x,
                  uint y)
{
    return 0;
}

int
overlapOmp(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2,
           uint height2, uint x, uint y)
{
    return 0;
}

int
overlapCuda(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2,
            uint height2, uint x, uint y)
{
    return 0;
}