//
// Created by f3m on 19/01/24.
//

#include "composition.cuh"

int copyMatrix(const unsigned char *mIn, unsigned char *mOut, uint widthI, uint heightI, uint widthO, uint x, uint y)
{

    unsigned char r;
    unsigned char g;
    unsigned char b;

    for (int i = 0; i < widthI; ++i)
    {
        for (int j = 0; j < heightI; ++j)
        {
            r = mIn[(i + j * widthI) * 3];
            g = mIn[(i + j * widthI) * 3 + 1];
            b = mIn[(i + j * widthI) * 3 + 2];

            uint xO = x + i;
            uint yO = y + j;

            mOut[(xO + yO * widthO) * 3] = r;
            mOut[(xO + yO * widthO) * 3 + 1] = g;
            mOut[(xO + yO * widthO) * 3 + 2] = b;
        }
    }

    return 0;
}

int parseCompositionArgs(char *args)
{
    char *img1 = strtok(args, " ");
    char *img2 = strtok(nullptr, " ");
    char *pathOut = strtok(nullptr, " ");
    //TODO: check meglio (crasha se ci sono troppi pochi valori)
    int side = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (img1 == nullptr || img2 == nullptr || pathOut == nullptr)
    {
        handle_error("Invalid arguments for composition function.\n");
    }

    //TODO: leggere le immagini in base alla loro estensione

    char *tpp = getStrFromKey((char *) "TPP");
    uint width1;
    uint height1;
    uint width2;
    uint height2;


    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img1_1 = loadPPM(img1, &width1, &height1);
        unsigned char *img2_1 = loadPPM(img2, &width2, &height2);
        compositionSerial(img1_1, img2_1, pathOut, width1, height1, width2, height2, side);
        free(img1_1);
        free(img2_1);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img1_1 = loadPPM(img1, &width1, &height1);
        unsigned char *img2_1 = loadPPM(img2, &width2, &height2);
        compositionOmp(img1_1, img2_1, pathOut, width1, height1, width2, height2, side);
        free(img1_1);
        free(img2_1);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img1_1 = loadPPM(img1, &width1, &height1);
        unsigned char *img2_1 = loadPPM(img2, &width2, &height2);
        compositionCuda(img1_1, img2_1, pathOut, width1, height1, width2, height2, side);
        free(img1_1);
        free(img2_1);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for composition function.\n");
    }
    free(tpp);
    return 0;
}


int compositionSerial(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2, uint height2, int side)
{
    uint widthOut = width1;
    uint heightOut = height1;
    switch (side)
    {
        case UP:
        case DOWN:
            heightOut += height2;
            break;
        case LEFT:
        case RIGHT:
            widthOut += width2;
            break;
        default:
        {
            handle_error("Parametro side non valido!\n");
        }
    }


    auto *imgOut = (unsigned char *) calloc(sizeof(unsigned char), widthOut * heightOut * 3);
    switch (side)
    {

        case UP:
            copyMatrix(img2, imgOut, width2, height2, widthOut, 0, 0);
            copyMatrix(img1, imgOut, width1, height1, widthOut, 0, height2);
            break;
        case DOWN:
            copyMatrix(img1, imgOut, width1, height1, widthOut, 0, 0);
            copyMatrix(img2, imgOut, width2, height2, widthOut, 0, height1);
            break;
        case LEFT:
            copyMatrix(img1, imgOut, width1, height1, widthOut, 0, 0);
            copyMatrix(img2, imgOut, width2, height2, widthOut, width1, 0);
            break;
        case RIGHT:
            copyMatrix(img2, imgOut, width2, height2, widthOut, 0, 0);
            copyMatrix(img1, imgOut, width1, height1, widthOut, width2, 0);
            break;
    }
    writePPM(pathOut, imgOut, widthOut, heightOut, "P6");

    free(imgOut);
    return 0;
}
int compositionOmp(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2, uint height2, int side)
{
    return 0;
}
int compositionCuda(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2, uint height2, int side)
{
    return 0;
}