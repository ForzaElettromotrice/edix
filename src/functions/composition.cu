//
// Created by f3m on 19/01/24.
//

#include "composition.cuh"

int copyMatrix(const unsigned char *mIn, unsigned char *mOut, uint widthI, uint heightI, uint widthO, uint channels1, uint channels2, uint x, uint y)
{
    //TODO: riparare il segfault
    for (int i = 0; i < widthI; ++i)
        for (int j = 0; j < heightI; ++j)
        {
            uint xO = x + i;
            uint yO = y + j;
            if (channels2 == 3 || channels1 == channels2)
            {
                for (int k = 0; k < channels1; ++k)
                {
                    mOut[(xO + yO * widthO) * channels1 + k] = mIn[(i + j * widthI) * channels2 + k];
                }
            } else
            {
                for (int k = 0; k < channels1; ++k)
                    mOut[(xO + yO * widthO) * channels1 + k] = mIn[i + j * widthI];
            }
        }

    return 0;
}

int copyMatrixOmp(const unsigned char *mIn, unsigned char *mOut, uint widthI, uint heightI, uint widthO, uint x, uint y, int nThread)
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    uint xO, yO, index;
#pragma omp parallel for num_threads(nThread) collapse(2) default(none) shared(mIn, mOut, widthI, heightI, widthO, x, y) private(r, g, b, xO, yO, index)
    for (int i = 0; i < widthI; ++i)
    {
        for (int j = 0; j < heightI; ++j)
        {
            r = mIn[(i + j * widthI) * 3];
            g = mIn[(i + j * widthI) * 3 + 1];
            b = mIn[(i + j * widthI) * 3 + 2];

            xO = x + i;
            yO = y + j;

            index = (xO + yO * widthO) * 3;

            mOut[index] = r;
            mOut[index + 1] = g;
            mOut[index + 2] = b;
        }
    }

    return 0;
}


unsigned char *compositionSerial(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight)
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
            E_Print(RED "Error: " RESET "Parametro side non valido!\n");
            return nullptr;
        }
    }


    uint oSize = widthOut * heightOut * (channels1 == 3 ? 3 : channels2);
    auto *imgOut = (unsigned char *) calloc(sizeof(unsigned char), oSize);

    switch (side)
    {
        case UP:
            copyMatrix(img2, imgOut, width2, height2, widthOut, channels1, channels2, 0, 0);
            copyMatrix(img1, imgOut, width1, height1, widthOut, channels1, channels2, 0, height2);
            break;
        case DOWN:
            copyMatrix(img1, imgOut, width1, height1, widthOut, channels1, channels2, 0, 0);
            copyMatrix(img2, imgOut, width2, height2, widthOut, channels1, channels2, 0, height1);
            break;
        case LEFT:
            copyMatrix(img1, imgOut, width1, height1, widthOut, channels1, channels2, 0, 0);
            copyMatrix(img2, imgOut, width2, height2, widthOut, channels1, channels2, width1, 0);
            break;
        case RIGHT:
            copyMatrix(img2, imgOut, width2, height2, widthOut, channels1, channels2, 0, 0);
            copyMatrix(img1, imgOut, width1, height1, widthOut, channels1, channels2, width2, 0);
            break;
    }

    *oWidth = widthOut;
    *oHeight = heightOut;

    return imgOut;
}
unsigned char *compositionOmp(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight, int nThread)
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
            E_Print(RED "Error: " RESET "Parametro side non valido!\n");
            return nullptr;
        }
    }
    auto *imgOut = (unsigned char *) calloc(sizeof(unsigned char), widthOut * heightOut * 3);
    switch (side)
    {

        case UP:
            copyMatrixOmp(img2, imgOut, width2, height2, widthOut, 0, 0, nThread);
            copyMatrixOmp(img1, imgOut, width1, height1, widthOut, 0, height2, nThread);
            break;
        case DOWN:
            copyMatrixOmp(img1, imgOut, width1, height1, widthOut, 0, 0, nThread);
            copyMatrixOmp(img2, imgOut, width2, height2, widthOut, 0, height1, nThread);
            break;
        case LEFT:
            copyMatrixOmp(img1, imgOut, width1, height1, widthOut, 0, 0, nThread);
            copyMatrixOmp(img2, imgOut, width2, height2, widthOut, width1, 0, nThread);
            break;
        case RIGHT:
            copyMatrixOmp(img2, imgOut, width2, height2, widthOut, 0, 0, nThread);
            copyMatrixOmp(img1, imgOut, width1, height1, widthOut, width2, 0, nThread);
            break;
    }

    *oWidth = widthOut;
    *oHeight = heightOut;

    return imgOut;
}
unsigned char *compositionCuda(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight)
{
    return nullptr;
}