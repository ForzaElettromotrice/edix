//
// Created by f3m on 06/02/24.
//

#include "scale.cuh"

int parseUpscaleArgs(char *args)
{
    char *pathIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int factor = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (pathIn == nullptr || pathOut == nullptr || factor == 0)
    {
        handle_error("usage " BOLD "funx upscale IN OUT FACTOR\n" RESET);
    }

    char *tpp = getStrFromKey((char *) "TPP");
    char *tup = getStrFromKey((char *) "TUP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(pathIn, &width, &height, &channels);

    uint oWidth = 0;
    uint oHeight = 0;
    unsigned char *imgOut;

    if (strcmp(tpp, "Serial") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = scaleSerialBilinear(img, width, height, channels, factor, true, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = scaleSerialBicubic(img, width, height, channels, factor, true, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = scaleOmpBilinear(img, width, height, channels, factor, true, &oWidth, &oHeight, 4);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = scaleOmpBicubic(img, width, height, channels, factor, true, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = scaleCudaBilinear(img, width, height, channels, factor, true, &oWidth, &oHeight, true);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = scaleCudaBicubic(img, width, height, channels, factor, true, &oWidth, &oHeight, true);
    } else
    {
        free(img);
        free(tpp);
        handle_error("Invalid TPP\n");
    }

    if (imgOut != nullptr)
    {
        writeImage(pathOut, imgOut, oWidth, oHeight, channels);
        free(imgOut);
    }
    free(img);
    free(tpp);
    return 0;
}
int parseDownscaleArgs(char *args)
{
    char *pathIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int factor = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (pathIn == nullptr || pathOut == nullptr || factor == 0)
    {
        handle_error("usage " BOLD "funx downscale IN OUT FACTOR\n" RESET);
    }

    char *tpp = getStrFromKey((char *) "TPP");
    char *tup = getStrFromKey((char *) "TUP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(pathIn, &width, &height, &channels);

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = scaleSerialBilinear(img, width, height, channels, factor, false, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = scaleSerialBicubic(img, width, height, channels, factor, false, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = scaleOmpBilinear(img, width, height, channels, factor, false, &oWidth, &oHeight, 4);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = scaleOmpBicubic(img, width, height, channels, factor, false, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = scaleCudaBilinear(img, width, height, channels, factor, false, &oWidth, &oHeight, false);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = scaleCudaBicubic(img, width, height, channels, factor, false, &oWidth, &oHeight, false);
    } else
    {
        free(tpp);
        handle_error("Invalid TPP\n");
    }

    if (oImg != nullptr)
    {
        writeImage(pathOut, oImg, oWidth, oHeight, channels);
        free(oImg);
    }

    free(img);
    free(tpp);
    return 0;
}


__host__ __device__ int bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta)
{
    return (int) ((1 - alpha) * (1 - beta) * p00 + (1 - alpha) * beta * p01 + alpha * (1 - beta) * p10 +
                  alpha * beta * p11);
}
__host__ __device__ void createSquare(unsigned char square[16][3], const unsigned char *img, int x, int y, uint width, uint height, uint channels)
{
    for (int i = -1; i < 3; ++i)
        for (int j = -1; j < 3; ++j)
        {
            for (int k = 0; k < channels; ++k)
            {

                if (x - i < 0 || y - j < 0 || x + i >= width || y + j >= height)
                {
                    square[(i + 1) + (j + 1) * 4][k] = img[channels * (x + y * width) + k];
                } else
                    square[(i + 1) + (j + 1) * 4][k] = img[channels * (x + i + (y + j) * width) + k];
            }
        }
}
__host__ __device__ double bicubicInterpolation(double A, double B, double C, double D, double t)
{
    double a = -A / 2.0f + (3.0f * B) / 2.0f - (3.0f * C) / 2.0f + D / 2.0f;
    double b = A - (5.0f * B) / 2.0f + 2.0f * C - D / 2.0f;
    double c = -A / 2.0f + C / 2.0f;
    double d = B;
    return a * t * t * t + b * t * t + c * t + d;
}

__global__ void scaleBilinear(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint oWidth, uint oHeight, uint channels, int factor, bool upscale)
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

        imgOut[(x + y * oWidth) * channels + k] = bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
    }
}
__global__ void scaleBilinearShared(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint oWidth, uint oHeight, uint channels, int factor)
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

        imgOut[(absX + absY * oWidth) * channels + k] = bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
    }
}

__global__ void scaleBicubic(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint oWidth, uint oHeight, uint channels, int factor, bool upscale)
{

    int x = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= oWidth && y >= oHeight)
        return;


    int i;
    int j;
    double alpha;
    double beta;
    unsigned char square[16][3];

    i = upscale ? x / factor : x * factor;
    j = upscale ? y / factor : y * factor;

    alpha = upscale ? ((double) x / factor) - i : 0.5;
    beta = upscale ? ((double) y / factor) - j : 0.5;

    createSquare(square, imgIn, i, j, iWidth, iHeight, channels);

    for (int k = 0; k < channels; k++)
    {
        double p1 = bicubicInterpolation(square[0][k], square[1][k], square[2][k], square[3][k], alpha);
        double p2 = bicubicInterpolation(square[4][k], square[5][k], square[6][k], square[7][k], alpha);
        double p3 = bicubicInterpolation(square[8][k], square[9][k], square[10][k], square[11][k], alpha);
        double p4 = bicubicInterpolation(square[12][k], square[13][k], square[14 + k][k], square[15][k], alpha);
        double p = bicubicInterpolation(p1, p2, p3, p4, beta);

        if (p > 255)
            p = 255;
        else if (p < 0)
            p = 0;

        imgOut[(x + y * oWidth) * channels + k] = (int) p;
    }
}
__global__ void scaleBicubicShared(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint oWidth, uint oHeight, uint channels, int factor)
{
    int absX = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int absY = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (absX >= oWidth || absY >= oHeight)
        return;

    int relX = (int) threadIdx.x;
    int relY = (int) threadIdx.y;

    uint sSize = ((uint) (8 + factor - 1) / factor) + 3;
    extern __shared__ unsigned char shared[];

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
    double alpha;
    double beta;
    unsigned char square[16][3];

    x = relX / factor;
    y = relY / factor;

    alpha = ((double) relX / factor) - x;
    beta = ((double) relY / factor) - y;


    createSquare(square, shared, x, y, sSize, sSize, channels);

    for (int k = 0; k < channels; k++)
    {
        double p1 = bicubicInterpolation(square[0][k], square[1][k], square[2][k], square[3][k], alpha);
        double p2 = bicubicInterpolation(square[4][k], square[5][k], square[6][k], square[7][k], alpha);
        double p3 = bicubicInterpolation(square[8][k], square[9][k], square[10][k], square[11][k], alpha);
        double p4 = bicubicInterpolation(square[12][k], square[13][k], square[14 + k][k], square[15][k], alpha);
        double p = bicubicInterpolation(p1, p2, p3, p4, beta);

        if (p > 255)
            p = 255;
        else if (p < 0)
            p = 0;

        imgOut[(absX + absY * oWidth) * channels + k] = (int) p;
    }
}

unsigned char *scaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight)
{
    *oWidth = upscale ? width * factor : width / factor;
    *oHeight = upscale ? height * factor : height / factor;

    uint oSize = *oWidth * *oHeight * channels;

    auto *imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
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


                imgOut[(i + j * *oWidth) * channels + k] = bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
            }
        }

    return imgOut;
}
unsigned char *scaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight, int nThreads)
{
    *oWidth = upscale ? width * factor : width / factor;
    *oHeight = upscale ? height * factor : height / factor;

    uint oSize = *oWidth * *oHeight * channels;

    auto *imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
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


                imgOut[(i + j * *oWidth) * channels + k] = bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
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

    //scaleBilinear
    dim3 gridSize = {(*oWidth + 7) / 8, (*oHeight + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    if (upscale && useShared)
    {
        size_t sharedDim = (size_t) pow((uint) ((double) (8 + factor - 1) / factor + 1), 2) * channels;
        scaleBilinearShared<<<gridSize, blockSize, sharedDim>>>(d_imgIn, d_imgOut, width, height, *oWidth, *oHeight, channels, factor);
    } else
        scaleBilinear<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, *oWidth, *oHeight, channels, factor, upscale);

    cudaMemcpy(h_imgOut, d_imgOut, oSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    munlock(h_imgIn, iSize * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    return h_imgOut;
}

unsigned char *scaleSerialBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight)
{
    *oWidth = upscale ? width * factor : width / factor;
    *oHeight = upscale ? height * factor : height / factor;

    uint oSize = *oWidth * *oHeight * channels;

    auto *imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Error while malloc!\n");
        return nullptr;
    }

    int x;
    int y;
    double alpha;
    double beta;
    unsigned char square[16][3];

    for (int i = 0; i < *oWidth; i++)
    {
        for (int j = 0; j < *oHeight; ++j)
        {
            x = upscale ? i / factor : i * factor;
            y = upscale ? j / factor : j * factor;

            alpha = upscale ? ((double) i / factor) - x : 0.5;
            beta = upscale ? ((double) j / factor) - y : 0.5;

            createSquare(square, imgIn, x, y, width, height, channels);

            for (int k = 0; k < channels; k++)
            {
                double p1 = bicubicInterpolation(square[0][k], square[1][k], square[2][k], square[3][k], alpha);
                double p2 = bicubicInterpolation(square[4][k], square[5][k], square[6][k], square[7][k], alpha);
                double p3 = bicubicInterpolation(square[8][k], square[9][k], square[10][k], square[11][k], alpha);
                double p4 = bicubicInterpolation(square[12][k], square[13][k], square[14 + k][k], square[15][k], alpha);
                double p = bicubicInterpolation(p1, p2, p3, p4, beta);

                if (p > 255)
                    p = 255;
                else if (p < 0)
                    p = 0;

                imgOut[(i + j * *oWidth) * channels + k] = (int) p;
            }
        }
    }


    return imgOut;
}
unsigned char *scaleOmpBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight, int nThreads)
{
    *oWidth = upscale ? width * factor : width / factor;
    *oHeight = upscale ? height * factor : height / factor;

    uint oSize = *oWidth * *oHeight * channels;

    auto *imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Error while malloc!\n");
        return nullptr;
    }

    int x;
    int y;
    double alpha;
    double beta;
    unsigned char square[16][3];

#pragma omp parallel for num_threads(nThreads) collapse(2) schedule(static) default(none) shared(oWidth, oHeight, factor, upscale, imgIn, imgOut, width, height, channels) private(x, y, alpha, beta, square)
    for (int i = 0; i < *oWidth; i++)
        for (int j = 0; j < *oHeight; ++j)
        {
            x = upscale ? i / factor : i * factor;
            y = upscale ? j / factor : j * factor;

            alpha = upscale ? ((double) i / factor) - x : 0.5;
            beta = upscale ? ((double) j / factor) - y : 0.5;

            createSquare(square, imgIn, x, y, width, height, channels);

            for (int k = 0; k < channels; k++)
            {
                double p1 = bicubicInterpolation(square[0][k], square[1][k], square[2][k], square[3][k], alpha);
                double p2 = bicubicInterpolation(square[4][k], square[5][k], square[6][k], square[7][k], alpha);
                double p3 = bicubicInterpolation(square[8][k], square[9][k], square[10][k], square[11][k], alpha);
                double p4 = bicubicInterpolation(square[12][k], square[13][k], square[14 + k][k], square[15][k], alpha);
                double p = bicubicInterpolation(p1, p2, p3, p4, beta);

                if (p > 255)
                    p = 255;
                else if (p < 0)
                    p = 0;

                imgOut[(i + j * *oWidth) * channels + k] = (int) p;
            }
        }


    return imgOut;
}
unsigned char *scaleCudaBicubic(const unsigned char *h_imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight, bool useShared)
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

    //scaleBilinear
    dim3 gridSize = {(*oWidth + 7) / 8, (*oHeight + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    if (upscale && useShared)
    {
        size_t sharedDim = (size_t) pow((uint) ((double) (8 + factor - 1) / factor + 3), 2) * channels;
        scaleBicubicShared<<<gridSize, blockSize, sharedDim>>>(d_imgIn, d_imgOut, width, height, *oWidth, *oHeight, channels, factor);
    } else
        scaleBicubic<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, height, *oWidth, *oHeight, channels, factor, upscale);

    cudaMemcpy(h_imgOut, d_imgOut, oSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    munlock(h_imgIn, iSize * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    return h_imgOut;
}