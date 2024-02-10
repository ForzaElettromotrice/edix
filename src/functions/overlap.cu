//
// Created by f3m on 19/01/24.
//

#include "overlap.cuh"

__global__ void overlap(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint channels1, uint oWidth, uint oHeight, uint x, uint y)
{
    int relX = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int relY = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (relX >= iWidth || relY >= iHeight || x + relX >= oWidth || y + relY >= oHeight)
        return;

    for (int k = 0; k < channels1; ++k)
        imgOut[(x + relX + (y + relY) * oWidth) * channels1 + k] = imgIn[(relX + relY * iWidth) * channels1 + k];
}

unsigned char *overlapSerial(const unsigned char *imgIn1, const unsigned char *imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight)
{
    *oWidth = width1;
    *oHeight = height1;
    uint oSize = *oWidth * *oHeight * channels1;

    auto *imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore nell' allocazione della memoria!\n");
        return nullptr;
    }

    memcpy(imgOut, imgIn1, oSize * sizeof(unsigned char));

    for (int i = 0; i < width2; ++i)
        for (int j = 0; j < height2; ++j)
        {
            if (x + i >= width1 || y + j >= height1)
                break;
            for (int k = 0; k < channels2; ++k)
                imgOut[(x + i + (y + j) * width1) * channels1 + k] = imgIn2[(i + j * width2) * channels2 + k];
        }

    return imgOut;
}
unsigned char *overlapOmp(const unsigned char *imgIn1, const unsigned char *imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight, int nThreads)
{
    *oWidth = width1;
    *oHeight = height1;
    uint oSize = *oWidth * *oHeight * channels1;

    auto *imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore nell' allocazione della memoria!\n");
        return nullptr;
    }

    memcpy(imgOut, imgIn1, oSize * sizeof(unsigned char));

#pragma omp parallel for num_threads(nThreads) collapse(2) schedule(static) default(none) shared(width2, height2, channels2, width1, height1, channels1, imgOut, imgIn2, x, y)
    for (int i = 0; i < width2; ++i)
        for (int j = 0; j < height2; ++j)
        {
            if (x + i >= width1 || y + j >= height1)
                continue;
            for (int k = 0; k < channels2; ++k)
                imgOut[(x + i + (y + j) * width1) * channels1 + k] = imgIn2[(i + j * width2) * channels2 + k];
        }

    return imgOut;
}
unsigned char *overlapCuda(const unsigned char *h_imgIn1, const unsigned char *h_imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight)
{
    *oWidth = width1;
    *oHeight = height1;
    uint oSize = *oWidth * *oHeight * channels1;
    uint iSize2 = width2 * height2 * channels2;

    auto *h_imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (h_imgOut == nullptr)
    {
        E_Print("Errore nell' allocazione della memoria!\n");
        return nullptr;
    }
    mlock(h_imgIn1, oSize * sizeof(unsigned char));
    mlock(h_imgIn2, iSize2 * sizeof(unsigned char));


    unsigned char *d_imgIn2;
    unsigned char *d_imgOut;
    cudaMalloc(&d_imgIn2, iSize2 * sizeof(unsigned char));
    cudaMalloc(&d_imgOut, oSize * sizeof(unsigned char));


    cudaMemcpy(d_imgIn2, h_imgIn2, iSize2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgOut, h_imgIn1, oSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 grisSize = {(width2 + 7) / 8, (height2 + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    overlap<<<grisSize, blockSize>>>(d_imgIn2, d_imgOut, width2, height2, channels1, *oWidth, *oHeight, x, y);


    cudaMemcpy(h_imgOut, d_imgOut, oSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    munlock(h_imgIn1, oSize * sizeof(unsigned char));
    munlock(h_imgIn2, iSize2 * sizeof(unsigned char));
    cudaFree(d_imgIn2);
    cudaFree(d_imgOut);

    return h_imgOut;
}