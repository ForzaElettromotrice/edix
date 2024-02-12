//
// Created by f3m on 19/01/24.
//

#include "colorfilter.cuh"


__global__ void colorFilter(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, uint channels, int r, int g, int b, uint squareTolerance)
{
    int x = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= width || y >= height)
        return;

    int diff[] = {0, 0, 0};
    int RGB[] = {r, g, b};
    uint squareDistance;

    for (int k = 0; k < channels; ++k)
        diff[k] = imgIn[(x + y * width) * channels + k] - RGB[k];

    squareDistance = (uint) (pow(diff[0], 2) + pow(diff[1], 2) + pow(diff[2], 2));

    if (squareDistance > squareTolerance)
        for (int k = 0; k < channels; ++k)
        {
            int val = (imgIn[(x + y * width) * channels + k] + diff[k]) / 2;
            if (val > 255)
                val = 255;
            if (val < 0)
                val = 0;

            imgOut[(x + y * width) * channels + k] = val;
        }

    else
        for (int k = 0; k < channels; ++k)
            imgOut[(x + y * width) * channels + k] = imgIn[(x + y * width) * channels + k];

}


unsigned char *colorFilterSerial(const unsigned char *imgIn, uint width, uint height, uint channels, int r, int g, int b, uint tolerance, uint *oWidth, uint *oHeight)
{
    uint oSize = width * height * channels;
    *oWidth = width;
    *oHeight = height;

    auto *imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }

    int diff[] = {0, 0, 0};
    int RGB[] = {r, g, b};
    int squareTolerance = (int) (pow(tolerance, 2));
    uint squareDistance;
    int test1 = 0, test2 = 0;

    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
        {
            for (int k = 0; k < channels; ++k)
                diff[k] = imgIn[(i + j * width) * channels + k] - RGB[k];

            squareDistance = (uint) (pow(diff[0], 2) + pow(diff[1], 2) + pow(diff[2], 2));

            if (squareDistance > squareTolerance)
                for (int k = 0; k < channels; ++k)
                {
                    int val = (imgIn[(i + j * width) * channels + k] + diff[k]) / 2;
                    if (val > 255)
                        val = 255;
                    if (val < 0)
                        val = 0;

                    imgOut[(i + j * width) * channels + k] = val;
                }
            else
                for (int k = 0; k < channels; ++k)
                    imgOut[(i + j * width) * channels + k] = imgIn[(i + j * width) * channels + k];
        }
    return imgOut;
}
unsigned char *colorFilterOmp(const unsigned char *imgIn, uint width, uint height, uint channels, int r, int g, int b, uint tolerance, uint *oWidth, uint *oHeight, int nThreads)
{
    uint oSize = width * height * channels;
    *oWidth = width;
    *oHeight = height;

    auto *imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }

    int diff[] = {0, 0, 0};
    int RGB[] = {r, g, b};
    int squareTolerance = (int) (pow(tolerance, 2));
    uint squareDistance;

#pragma omp parallel for num_threads(nThreads) collapse(2) schedule(static) default(none) shared(width, height, channels, imgIn, imgOut, RGB, squareTolerance) private(diff, squareDistance)
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
        {
            for (int k = 0; k < channels; ++k)
                diff[k] = imgIn[(i + j * width) * channels + k] - RGB[k];

            squareDistance = (uint) (pow(diff[0], 2) + pow(diff[1], 2) + pow(diff[2], 2));

            if (squareDistance > squareTolerance)
                for (int k = 0; k < channels; ++k)
                {
                    int val = (imgIn[(i + j * width) * channels + k] + diff[k]) / 2;
                    if (val > 255)
                        val = 255;
                    if (val < 0)
                        val = 0;

                    imgOut[(i + j * width) * channels + k] = val;
                }
            else
                for (int k = 0; k < channels; ++k)
                    imgOut[(i + j * width) * channels + k] = imgIn[(i + j * width) * channels + k];
        }
    return imgOut;
}
unsigned char *colorFilterCuda(const unsigned char *h_imgIn, uint width, uint height, uint channels, int r, int g, int b, uint tolerance, uint *oWidth, uint *oHeight)
{
    uint oSize = width * height * channels;
    *oWidth = width;
    *oHeight = height;
    int squareTolerance = (int) (pow(tolerance, 2));

    auto *h_imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (h_imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }
    mlock(h_imgIn, oSize * sizeof(unsigned char));

    unsigned char *d_imgIn;
    unsigned char *d_imgOut;
    cudaMalloc(&d_imgIn, oSize * sizeof(unsigned char));
    cudaMalloc(&d_imgOut, oSize * sizeof(unsigned char));

    cudaMemcpy(d_imgIn, h_imgIn, oSize * sizeof(unsigned char), cudaMemcpyHostToDevice);


    dim3 gridSize = {(width + 7) / 8, (height + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};

    colorFilter<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, height, channels, r, g, b, squareTolerance);

    cudaMemcpy(h_imgOut, d_imgOut, oSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    munlock(h_imgOut, oSize * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    return h_imgOut;
}