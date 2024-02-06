//
// Created by f3m on 06/02/24.
//

#include "scale.cuh"


int h_bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta)
{
    return (int) ((1 - alpha) * (1 - beta) * p00 + (1 - alpha) * beta * p01 + alpha * (1 - beta) * p10 +
                  alpha * beta * p11);
}
__device__ int d_bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta)
{
    return (int) ((1 - alpha) * (1 - beta) * p00 + (1 - alpha) * beta * p01 + alpha * (1 - beta) * p10 +
                  alpha * beta * p11);
}

__global__ void scale(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint oWidth, uint oHeight, uint channels, int factor, bool upscale)
{
    int x = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= oWidth || y >= oHeight)
        return;

    int i;
    int j;
    int p00;
    int p01;
    int p10;
    int p11;
    double alpha;
    double beta;

    i = upscale ? x / factor : x * factor;
    j = upscale ? y / factor : y * factor;
    alpha = upscale ? ((double) x / factor) - i : 0.5;
    beta = upscale ? ((double) y / factor) - j : 0.5;

    for (int k = 0; k < channels; k++)
    {
        p00 = imgIn[(i + j * iWidth) * channels + k];
        p01 = imgIn[(i + 1 + j * iWidth) * channels + k];
        p10 = imgIn[(i + (j + 1) * iWidth) * channels + k];
        p11 = imgIn[(i + 1 + (j + 1) * iWidth) * channels + k];

        imgOut[(x + y * oWidth) * channels + k] = d_bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
    }
}
__global__ void scaleShared(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint oWidth, uint oHeight, uint channels, int factor)
{
    int absX = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int absY = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (absX >= oWidth || absY >= oHeight)
        return;

    int relX = (int) threadIdx.x;
    int relY = (int) threadIdx.y;

    uint sSize = ((uint) (8 + factor - 1) / factor) + 1;
    extern __shared__ unsigned char shared[];

    uint oldX = absX / factor;
    uint oldY = absY / factor;


    if (relX < sSize && relY < sSize)
    {
        int sX = (absX - relX) / factor + relX;
        int sY = (absY - relY) / factor + relY;
        if (sX < iWidth && sY < iHeight)
            for (int k = 0; k < channels; ++k)
                shared[(relX + relY * sSize) * channels + k] = imgIn[(sX + sY * iWidth) * channels + k];
    }
    __syncthreads();


    int x;
    int y;
    int p00;
    int p01;
    int p10;
    int p11;
    double alpha;
    double beta;

    x = relX / factor;
    y = relY / factor;
    alpha = ((double) relX / factor) - x;
    beta = ((double) relY / factor) - y;

    for (int k = 0; k < channels; k++)
    {
        p00 = shared[(x + y * sSize) * channels + k];
        p01 = oldX + x + 1 >= iWidth ? p00 : shared[(x + 1 + y * sSize) * channels + k];
        p10 = oldY + y + 1 >= iHeight ? p00 : shared[(x + (y + 1) * sSize) * channels + k];
        p11 = oldX + x + 1 >= iWidth || oldY + y + 1 >= iHeight ? p00 : shared[(x + 1 + (y + 1) * sSize) * channels + k];

        imgOut[(absX + absY * oWidth) * channels + k] = (int) ((1 - alpha) * (1 - beta) * p00 + (1 - alpha) * beta * p01 + alpha * (1 - beta) * p10 + alpha * beta * p11);
    }
}

unsigned char *scaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight)
{
    *oWidth = upscale ? width * factor : width / factor;
    *oHeight = upscale ? height * factor : height / factor;

    auto *imgOut = (unsigned char *) malloc((*oWidth * *oHeight * channels) * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Error while malloc!\n");
        return nullptr;
    }

    int x;
    int y;
    int p00;
    int p01;
    int p10;
    int p11;
    double alpha;
    double beta;

    for (int i = 0; i < *oWidth; ++i)
        for (int j = 0; j < *oHeight; ++j)
        {
            x = upscale ? i / factor : i * factor;
            y = upscale ? j / factor : j * factor;

            alpha = upscale ? ((double) i / factor) - x : 0.5;
            beta = upscale ? ((double) j / factor) - y : 0.5;

            for (int k = 0; k < channels; ++k)
            {
                p00 = imgIn[(x + y * width) * channels + k];
                p01 = x + 1 >= width ? p00 : imgIn[(x + 1 + y * width) * channels + k];
                p10 = y + 1 >= height ? p00 : imgIn[(x + (y + 1) * width) * channels + k];
                p11 = x + 1 >= width || y + 1 >= height ? p00 : imgIn[(x + 1 + (y + 1) * width) * channels + k];


                imgOut[(i + j * *oWidth) * channels + k] = h_bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
            }
        }

    return imgOut;
}
unsigned char *scaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight, int nThreads)
{
    *oWidth = upscale ? width * factor : width / factor;
    *oHeight = upscale ? height * factor : height / factor;

    auto *imgOut = (unsigned char *) malloc((*oWidth * *oHeight * channels) * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Error while malloc!\n");
        return nullptr;
    }

    int x;
    int y;
    int p00;
    int p01;
    int p10;
    int p11;
    double alpha;
    double beta;

#pragma omp parallel for num_threads(nThreads) collapse(2) schedule(static) default(none) shared(oWidth, oHeight, factor, upscale, imgIn, imgOut, width, height, channels) private(x, y, alpha, beta, p00, p01, p10, p11)
    for (int i = 0; i < *oWidth; ++i)
        for (int j = 0; j < *oHeight; ++j)
        {
            x = upscale ? i / factor : i * factor;
            y = upscale ? j / factor : j * factor;

            alpha = upscale ? ((double) i / factor) - x : 0.5;
            beta = upscale ? ((double) j / factor) - y : 0.5;

            for (int k = 0; k < channels; ++k)
            {
                p00 = imgIn[(x + y * width) * channels + k];
                p01 = x + 1 >= width ? p00 : imgIn[(x + 1 + y * width) * channels + k];
                p10 = y + 1 >= height ? p00 : imgIn[(x + (y + 1) * width) * channels + k];
                p11 = x + 1 >= width || y + 1 >= height ? p00 : imgIn[(x + 1 + (y + 1) * width) * channels + k];


                imgOut[(i + j * *oWidth) * channels + k] = h_bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
            }
        }

    return imgOut;
}
unsigned char *scaleCudaBilinear(const unsigned char *h_imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight, bool useShared)
{
    *oWidth = upscale ? width * factor : width / factor;
    *oHeight = upscale ? height * factor : height / factor;

    uint iSize = width * height * channels;
    uint oSize = *oWidth * *oHeight * channels;

    auto *h_imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (h_imgOut == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Error while malloc!\n");
        return nullptr;
    }
    mlock(h_imgIn, iSize * sizeof(unsigned char));

    unsigned char *d_imgIn;
    unsigned char *d_imgOut;
    cudaMalloc(&d_imgIn, iSize * sizeof(unsigned char));
    cudaMalloc(&d_imgOut, oSize * sizeof(unsigned char));

    cudaMemcpy(d_imgIn, h_imgIn, iSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //scale
    dim3 gridSize = {(*oWidth + 7) / 8, (*oHeight + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    if (upscale && useShared)
    {
        size_t sharedDim = (size_t) pow((uint) ((double) (8 + factor - 1) / factor + 2), 2) * channels;
        scaleShared<<<gridSize, blockSize, sharedDim>>>(d_imgIn, d_imgOut, width, height, *oWidth, *oHeight, channels, factor);
    } else
        scale<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, *oWidth, *oHeight, channels, factor, upscale);

    cudaMemcpy(h_imgOut, d_imgOut, oSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    munlock(h_imgIn, iSize * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    return h_imgOut;
}

