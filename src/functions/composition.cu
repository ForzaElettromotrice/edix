//
// Created by f3m on 19/01/24.
//

#include "composition.cuh"

int copyMatrix(const unsigned char *mIn, unsigned char *mOut, uint widthI, uint heightI, uint widthO, uint channels1, uint channels2, uint x, uint y)
{
    for (int i = 0; i < widthI; ++i)
        for (int j = 0; j < heightI; ++j)
        {
            uint xO = x + i;
            uint yO = y + j;
            for (int k = 0; k < channels1; ++k)
                mOut[(xO + yO * widthO) * channels1 + k] = mIn[(i + j * widthI) * channels2 + k];

        }

    return 0;
}
int copyMatrixOmp(const unsigned char *mIn, unsigned char *mOut, uint widthI, uint heightI, uint widthO, uint channels1, uint channels2, uint x, uint y, int nThread)
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
#pragma omp parallel for num_threads(nThread) collapse(2) default(none) shared(mIn, mOut, widthI, heightI, widthO, x, y, channels1, channels2) private(r, g, b)
    for (int i = 0; i < widthI; ++i)
        for (int j = 0; j < heightI; ++j)
        {
            uint xO = x + i;
            uint yO = y + j;
            for (int k = 0; k < channels1; ++k)
                mOut[(xO + yO * widthO) * channels1 + k] = mIn[(i + j * widthI) * channels2 + k];
        }

    return 0;
}
__global__ void copyMatrixCuda(const unsigned char *mIn, unsigned char *mOut, uint widthI, uint heightI, uint widthO, uint channels1, uint x, uint y)
{
    int relX = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int relY = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (relX >= widthI || relY >= heightI)
        return;
    for (int k = 0; k < channels1; ++k)
        mOut[(x + relX + (y + relY) * widthO) * channels1 + k] = mIn[(relX + relY * widthI) * channels1 + k];
}

unsigned char *compositionSerial(const unsigned char *imgIn1, const unsigned char *imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight)
{
    *oWidth = width1;
    *oHeight = height1;
    switch (side)
    {
        case UP:
        case DOWN:
            *oHeight += height2;
            break;
        case LEFT:
        case RIGHT:
            *oWidth += width2;
            break;
        default:
        {
            E_Print(RED "Error: " RESET "Parametro side non valido!\n");
            return nullptr;
        }
    }

    uint oSize = *oWidth * *oHeight * channels1;
    auto imgOut = (unsigned char *) calloc(oSize, sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }


    switch (side)
    {
        case UP:
            copyMatrix(imgIn2, imgOut, width2, height2, *oWidth, channels1, channels2, 0, 0);
            copyMatrix(imgIn1, imgOut, width1, height1, *oWidth, channels1, channels2, 0, height2);
            break;
        case DOWN:
            copyMatrix(imgIn1, imgOut, width1, height1, *oWidth, channels1, channels2, 0, 0);
            copyMatrix(imgIn2, imgOut, width2, height2, *oWidth, channels1, channels2, 0, height1);
            break;
        case LEFT:
            copyMatrix(imgIn1, imgOut, width1, height1, *oWidth, channels1, channels2, 0, 0);
            copyMatrix(imgIn2, imgOut, width2, height2, *oWidth, channels1, channels2, width1, 0);
            break;
        case RIGHT:
            copyMatrix(imgIn2, imgOut, width2, height2, *oWidth, channels1, channels2, 0, 0);
            copyMatrix(imgIn1, imgOut, width1, height1, *oWidth, channels1, channels2, width2, 0);
            break;
    }

    return imgOut;
}
unsigned char *compositionOmp(const unsigned char *imgIn1, const unsigned char *imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight, int nThreads)
{
    *oWidth = width1;
    *oHeight = height1;
    switch (side)
    {
        case UP:
        case DOWN:
            *oHeight += height2;
            break;
        case LEFT:
        case RIGHT:
            *oWidth += width2;
            break;
        default:
        {
            E_Print(RED "Error: " RESET "Parametro side non valido!\n");
            return nullptr;
        }
    }

    uint oSize = *oWidth * *oHeight * channels1;
    auto imgOut = (unsigned char *) calloc(oSize, sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }


    switch (side)
    {
        case UP:
            copyMatrixOmp(imgIn2, imgOut, width2, height2, *oWidth, channels1, channels2, 0, 0, nThreads);
            copyMatrixOmp(imgIn1, imgOut, width1, height1, *oWidth, channels1, channels2, 0, height2, nThreads);
            break;
        case DOWN:
            copyMatrixOmp(imgIn1, imgOut, width1, height1, *oWidth, channels1, channels2, 0, 0, nThreads);
            copyMatrixOmp(imgIn2, imgOut, width2, height2, *oWidth, channels1, channels2, 0, height1, nThreads);
            break;
        case LEFT:
            copyMatrixOmp(imgIn1, imgOut, width1, height1, *oWidth, channels1, channels2, 0, 0, nThreads);
            copyMatrixOmp(imgIn2, imgOut, width2, height2, *oWidth, channels1, channels2, width1, 0, nThreads);
            break;
        case RIGHT:
            copyMatrixOmp(imgIn2, imgOut, width2, height2, *oWidth, channels1, channels2, 0, 0, nThreads);
            copyMatrixOmp(imgIn1, imgOut, width1, height1, *oWidth, channels1, channels2, width2, 0, nThreads);
            break;
    }

    return imgOut;
}
unsigned char *compositionOmpAlternative(const unsigned char *imgIn1, const unsigned char *imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight, int nThreads)
{
    *oWidth = width1;
    *oHeight = height1;
    switch (side)
    {
        case UP:
        case DOWN:
            *oHeight += height2;
            break;
        case LEFT:
        case RIGHT:
            *oWidth += width2;
            break;
        default:
        {
            E_Print(RED "Error: " RESET "Parametro side non valido!\n");
            return nullptr;
        }
    }

    uint oSize = *oWidth * *oHeight * channels1;
    auto imgOut = (unsigned char *) calloc(oSize, sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }


    const unsigned char *copyImg1, *copyImg2;
    uint copyWidth1, copyWidth2;
    uint copyHeight1, copyHeight2;
    uint copyX2, copyY2;

    switch (side)
    {
        case UP:
            copyImg1 = imgIn2;
            copyWidth1 = width2;
            copyHeight1 = height2;

            copyImg2 = imgIn1;
            copyWidth2 = width1;
            copyHeight2 = height1;
            copyX2 = 0;
            copyY2 = height2;
            break;
        case DOWN:
            copyImg1 = imgIn1;
            copyWidth1 = width1;
            copyHeight1 = height1;

            copyImg2 = imgIn2;
            copyWidth2 = width2;
            copyHeight2 = height2;
            copyX2 = 0;
            copyY2 = height1;
            break;
        case LEFT:
            copyImg1 = imgIn1;
            copyWidth1 = width1;
            copyHeight1 = height1;

            copyImg2 = imgIn2;
            copyWidth2 = width2;
            copyHeight2 = height2;
            copyX2 = width1;
            copyY2 = 0;
            break;
        case RIGHT:
            copyImg1 = imgIn2;
            copyWidth1 = width2;
            copyHeight1 = height2;

            copyImg2 = imgIn1;
            copyWidth2 = width1;
            copyHeight2 = height1;
            copyX2 = width2;
            copyY2 = 0;
            break;
    }

#pragma omp parallel num_threads(2) default(none) shared(copyImg1, copyImg2, imgOut, copyWidth1, copyHeight1, copyWidth2, copyHeight2, oWidth, channels1, channels2, copyX2, copyY2, nThreads)
    {
        int id = omp_get_thread_num();
        if (id == 0)
            copyMatrixOmp(copyImg1, imgOut, copyWidth1, copyHeight1, *oWidth, channels1, channels2, 0, 0, nThreads / 2 - 1);
        else
            copyMatrixOmp(copyImg2, imgOut, copyWidth2, copyHeight2, *oWidth, channels1, channels2, copyX2, copyY2, nThreads / 2 - 1);
    }

    return imgOut;
}
unsigned char *compositionCuda(const unsigned char *h_imgIn1, const unsigned char *h_imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight)
{
    *oWidth = width1;
    *oHeight = height1;
    switch (side)
    {
        case UP:
        case DOWN:
            *oHeight += height2;
            break;
        case LEFT:
        case RIGHT:
            *oWidth += width2;
            break;
        default:
        {
            E_Print(RED "Error: " RESET "Parametro side non valido!\n");
            return nullptr;
        }
    }

    uint oSize = *oWidth * *oHeight * channels1;
    uint iSize1 = width1 * height1 * channels1;
    uint iSize2 = width2 * height2 * channels2;

    auto h_imgOut = (unsigned char *) calloc(oSize, sizeof(unsigned char));
    if (h_imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }
    mlock(h_imgIn1, iSize1 * sizeof(unsigned char));
    mlock(h_imgIn2, iSize2 * sizeof(unsigned char));

    unsigned char *d_imgIn1;
    unsigned char *d_imgIn2;
    unsigned char *d_imgOut;
    cudaMalloc(&d_imgIn1, iSize1 * sizeof(unsigned char));
    cudaMalloc(&d_imgIn2, iSize2 * sizeof(unsigned char));
    cudaMalloc(&d_imgOut, oSize * sizeof(unsigned char));

    cudaMemcpy(d_imgIn1, h_imgIn1, iSize1 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgIn2, h_imgIn2, iSize2 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 gridSize1 = {(width1 + 7) / 8, (height1 + 7) / 8, 1};
    dim3 gridSize2 = {(width2 + 7) / 8, (height2 + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};

    switch (side)
    {
        case UP:
            copyMatrixCuda<<<gridSize2, blockSize>>>(d_imgIn2, d_imgOut, width2, height2, *oWidth, channels1, 0, 0);
            copyMatrixCuda<<<gridSize1, blockSize>>>(d_imgIn1, d_imgOut, width1, height1, *oWidth, channels1, 0, height2);
            break;
        case DOWN:
            copyMatrixCuda<<<gridSize1, blockSize>>>(d_imgIn1, d_imgOut, width1, height1, *oWidth, channels1, 0, 0);
            copyMatrixCuda<<<gridSize2, blockSize>>>(d_imgIn2, d_imgOut, width2, height2, *oWidth, channels1, 0, height1);
            break;
        case LEFT:
            copyMatrixCuda<<<gridSize1, blockSize>>>(d_imgIn1, d_imgOut, width1, height1, *oWidth, channels1, 0, 0);
            copyMatrixCuda<<<gridSize2, blockSize>>>(d_imgIn2, d_imgOut, width2, height2, *oWidth, channels1, width1, 0);
            break;
        case RIGHT:
            copyMatrixCuda<<<gridSize2, blockSize>>>(d_imgIn2, d_imgOut, width2, height2, *oWidth, channels1, 0, 0);
            copyMatrixCuda<<<gridSize1, blockSize>>>(d_imgIn1, d_imgOut, width1, height1, *oWidth, channels1, width2, 0);
            break;
    }

    cudaMemcpy(h_imgOut, d_imgOut, oSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    munlock(h_imgIn1, iSize1 * sizeof(unsigned char));
    munlock(h_imgIn2, iSize2 * sizeof(unsigned char));
    cudaFree(d_imgIn1);
    cudaFree(d_imgIn2);
    cudaFree(d_imgOut);

    return h_imgOut;
}
