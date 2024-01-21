//
// Created by f3m on 19/01/24.
//

#include "functions.cuh"

int parseOverlapArgs(char *args)
{
    char *img1 = strtok(args, " ");
    char *img2 = strtok(nullptr, " ");
    char *pathOut = strtok(nullptr, " ");
    //TODO: controllare se x e y sono stringhe o meno
    uint x = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);
    uint y = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);

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
        free(img1_1);
        free(img2_1);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img1_1 = loadPPM(img1, &width1, &height1);
        unsigned char *img2_1 = loadPPM(img2, &width2, &height2);
        overlapOmp(img1_1, img2_1, pathOut, width1, height1, width2, height2, x, y);
        free(img1_1);
        free(img2_1);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img1_1 = loadPPM(img1, &width1, &height1);
        unsigned char *img2_1 = loadPPM(img2, &width2, &height2);
        overlapCuda(img1_1, img2_1, pathOut, width1, height1, width2, height2, x, y);
        free(img1_1);
        free(img2_1);
    } else
    {
        free(tpp);
        handle_error("Invalid TPP\n");
    }
    free(tpp);
    return 0;
}

int overlapSerial(unsigned char *img1, const unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2, uint height2, uint x, uint y)
{
    if (x + width2 > width1 || y + height2 > height1)
    {
        handle_error("La secondo immagine è troppo grande per essere inserita li!\n");
    }
    for (int i = 0; i < width2; i++)
        for (int j = 0; j < height2; j++)
        {
            img1[3 * (x + i + (y + j) * width1)] = img2[3 * (i + j * width2)];
            img1[3 * (x + i + (y + j) * width1) + 1] = img2[3 * (i + j * width2) + 1];
            img1[3 * (x + i + (y + j) * width1) + 2] = img2[3 * (i + j * width2) + 2];
        }

    writePPM(pathOut, img1, width1, height1, "P6");
    return 0;
}
int overlapOmp(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2, uint height2, uint x, uint y)
{
    return 0;
}
int overlapCuda(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2, uint height2, uint x, uint y)
{
    return 0;
}