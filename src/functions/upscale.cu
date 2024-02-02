//
// Created by f3m on 19/01/24.
//

#include "upscale.cuh"

int bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta)
{
    return (int) ((1 - alpha) * (1 - beta) * p00 + (1 - alpha) * beta * p01 + alpha * (1 - beta) * p10 +
                  alpha * beta * p11);
}
double cubicInterpolate(double A, double B, double C, double D, double t)
{
    double a = -A / 2.0f + (3.0f * B) / 2.0f - (3.0f * C) / 2.0f + D / 2.0f;
    double b = A - (5.0f * B) / 2.0f + 2.0f * C - D / 2.0f;
    double c = -A / 2.0f + C / 2.0f;
    double d = B;
    return a * t * t * t + b * t * t + c * t + d;
//    return A + 0.5 * t * (C - A + t * (2.0 * A - 5.0 * B + 4.0 * C - D + t * (3.0 * (B - C) + D - A)));
}
void createSquare(unsigned char square[16][3], const unsigned char *img, int x, int y, uint width, uint height)
{
    for (int i = -1; i < 3; ++i)
    {
        for (int j = -1; j < 3; ++j)
        {
            if (x - i < 0 || y - j < 0 || x + i >= width || y + j >= height)
                continue;
            int r = img[3 * (x + i + (y + j) * width)];
            int g = img[3 * (x + i + (y + j) * width) + 1];
            int b = img[3 * (x + i + (y + j) * width) + 2];
            square[(i + 1) + (j + 1) * 3][0] = r;
            square[(i + 1) + (j + 1) * 3][1] = g;
            square[(i + 1) + (j + 1) * 3][2] = b;
        }
    }
}

__global__ void bilinearUpscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, int factor)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    uint widthO = width * factor;
    uint heightO = height * factor;

    uint idx = x + y * widthO;

    int i;
    int j;
    int p00;
    int p01;
    int p10;
    int p11;
    double alpha;
    double beta;

    if (idx < widthO * heightO * 3)
    {
        i = x / factor;
        j = y / factor;
        alpha = ((double) x / factor) - i;
        beta = ((double) y / factor) - j;

        for (int k = 0; k < 3; k++)
        {
            p00 = imgIn[(i + j * width) * 3 + k];
            p01 = imgIn[(i + 1 + j * width) * 3 + k];
            p10 = imgIn[(i + (j + 1) * width) * 3 + k];
            p11 = imgIn[(i + 1 + (j + 1) * width) * 3 + k];

            imgOut[(idx * 3) + k] = (int) ((1 - alpha) * (1 - beta) * p00 + (1 - alpha) *
                                                                            beta * p01 + alpha * (1 - beta) * p10 +
                                           alpha * beta * p11);
        }
    }
}
__global__ void bicubicUpscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, int factor, dim3 gridSize, dim3 blockSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    uint widthO = width * factor;
    uint heightO = height * factor;

    uint idx = x + y * widthO;

    int i;
    int j;
    double alpha;
    double beta;
    unsigned char square[16][3];
    int pixel[3];

    if (idx < widthO * heightO * 3)
    {
        i = x / factor;
        j = y / factor;

        alpha = ((double) x / factor) - i;
        beta = ((double) y / factor) - j;


        for (int k = -1; k < 3; ++k)
        {
            for (int z = -1; z < 3; ++z)
            {
                if (i - k < 0 || y - z < 0 || i + k >= width || j + z >= height)
                    continue;
                int r = imgIn[3 * (i + k + (j + z) * width)];
                int g = imgIn[3 * (i + k + (j + z) * width) + 1];
                int b = imgIn[3 * (i + k + (j + z) * width) + 2];
                square[(k + 1) + (z + 1) * 3][0] = r;
                square[(k + 1) + (z + 1) * 3][1] = g;
                square[(k + 1) + (z + 1) * 3][2] = b;
            }
        }

        for (int k = 0; k < 3; k++)
        {

            double a = -square[0][0 + k] / 2.0f + (3.0f * square[1][0 + k]) / 2.0f - (3.0f * square[2][0 + k]) / 2.0f + square[3][0 + k] / 2.0f;
            double b = square[0][0 + k] - (5.0f * square[1][0 + k]) / 2.0f + 2.0f * square[2][0 + k] - square[3][0 + k] / 2.0f;
            double c = -square[0][0 + k] / 2.0f + square[2][0 + k] / 2.0f;
            double d = square[1][0 + k];
            double p1 = a * alpha * alpha * alpha + b * alpha * alpha + c * alpha + d;


            a = -square[4][0 + k] / 2.0f + (3.0f * square[5][0 + k]) / 2.0f - (3.0f * square[6][0 + k]) / 2.0f + square[7][0 + k] / 2.0f;
            b = square[4][0 + k] - (5.0f * square[5][0 + k]) / 2.0f + 2.0f * square[6][0 + k] - square[7][0 + k] / 2.0f;
            c = -square[4][0 + k] / 2.0f + square[6][0 + k] / 2.0f;
            d = square[5][0 + k];
            double p2 = a * alpha * alpha * alpha + b * alpha * alpha + c * alpha + d;


            a = -square[8][0 + k] / 2.0f + (3.0f * square[9][0 + k]) / 2.0f - (3.0f * square[10][0 + k]) / 2.0f + square[11][0 + k] / 2.0f;
            b = square[8][0 + k] - (5.0f * square[9][0 + k]) / 2.0f + 2.0f * square[10][0 + k] - square[11][0 + k] / 2.0f;
            c = -square[8][0 + k] / 2.0f + square[10][0 + k] / 2.0f;
            d = square[9][0 + k];
            double p3 = a * alpha * alpha * alpha + b * alpha * alpha + c * alpha + d;


            a = -square[12][0 + k] / 2.0f + (3.0f * square[13][0 + k]) / 2.0f - (3.0f * square[14 + k][0 + k]) / 2.0f + square[15][0 + k] / 2.0f;
            b = square[12][0 + k] - (5.0f * square[13][0 + k]) / 2.0f + 2.0f * square[14 + k][0 + k] - square[15][0 + k] / 2.0f;
            c = -square[12][0 + k] / 2.0f + square[14 + k][0 + k] / 2.0f;
            d = square[13][0 + k];
            double p4 = a * alpha * alpha * alpha + b * alpha * alpha + c * alpha + d;


            a = -p1 / 2.0f + (3.0f * p2) / 2.0f - (3.0f * p3) / 2.0f + p4 / 2.0f;
            b = p1 - (5.0f * p2) / 2.0f + 2.0f * p3 - p4 / 2.0f;
            c = -p1 / 2.0f + p3 / 2.0f;
            d = p2;
            double p = a * beta * beta * beta + b * beta * beta + c * beta + d;

            if (p > 255)
                p = 255;
            else if (p < 0)
                p = 0;

            pixel[k] = (int) p;

        }

        imgOut[idx * 3] = pixel[0];
        imgOut[(idx * 3) + 1] = pixel[1];
        imgOut[(idx * 3) + 2] = pixel[2];
    }
}


int parseUpscaleArgs(char *args)
{
    char *pathIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int factor = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (pathIn == nullptr || pathOut == nullptr || factor == 0)
    {
        handle_error("Invalid arguments for upscale\n");
    }

    char *tpp = getStrFromKey((char *) "TPP");
    char *tup = getStrFromKey((char *) "TUP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(pathIn, &width, &height, &channels);

    uint oWidth;
    uint oHeight;
    unsigned char *imgOut;

    if (strcmp(tpp, "Serial") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = upscaleSerialBilinear(img, width, height, channels, factor, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = upscaleSerialBicubic(img, width, height, channels, factor, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = upscaleOmpBilinear(img, width, height, channels, factor, &oWidth, &oHeight, 4);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = upscaleOmpBicubic(img, width, height, channels, factor, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = upscaleCudaBilinear(img, width, height, channels, factor, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = upscaleCudaBicubic(img, width, height, channels, factor, &oWidth, &oHeight);
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

unsigned char *upscaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{

    uint widthO = width * factor;
    uint heightO = height * factor;
    auto *imgOut = (unsigned char *) malloc((widthO * heightO * 3) * sizeof(unsigned char));
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

    for (int i = 0; i < widthO; ++i)
    {
        for (int j = 0; j < heightO; ++j)
        {
            x = i / factor;
            y = j / factor;
            alpha = ((double) i / factor) - x;
            beta = ((double) j / factor) - y;

            for (int k = 0; k < channels; ++k)
            {
                //TODO: se sbordi, usa lo stesso pixel
                p00 = imgIn[(x + y * width) * 3 + k];
                p01 = imgIn[(x + 1 + y * width) * 3 + k];
                p10 = imgIn[(x + (y + 1) * width) * 3 + k];
                p11 = imgIn[(x + 1 + (y + 1) * width) * 3 + k];


                imgOut[(i + j * widthO) * 3 + k] = bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
            }
        }
    }

    *oWidth = widthO;
    *oHeight = heightO;

    return imgOut;
}
unsigned char *upscaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread)
{
    return nullptr;
}
unsigned char *upscaleCudaBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{
    uint wf, hf;

    //host   
    wf = width * factor;
    hf = height * factor;

    uint iSize = width * height * 3;
    uint iSizeO = wf * hf * 3;

    auto h_imgOut = (unsigned char *) malloc(iSizeO * sizeof(unsigned char));
    if (h_imgOut == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Errore nell'allocazione della memoria\n");
        munlock(imgIn, width * height * 3 * sizeof(unsigned char));
        return nullptr;
    }
    mlock(imgIn, iSize * sizeof(unsigned char));

    //device
    unsigned char *d_imgIn;
    unsigned char *d_imgOut;
    cudaMalloc(&d_imgIn, iSize * sizeof(unsigned char));
    cudaMalloc(&d_imgOut, iSizeO * sizeof(unsigned char));

    //copy
    cudaMemcpy(d_imgIn, imgIn, iSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //upscale
    dim3 gridSize = {(wf + 7) / 8, (hf + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    bilinearUpscaleCUDA<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, height, factor);

    //copy back
    cudaMemcpy(h_imgOut, d_imgOut, iSizeO * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //free
    munlock(imgIn, iSize * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    *oWidth = wf;
    *oHeight = hf;

    return h_imgOut;
}

unsigned char *upscaleSerialBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{

    uint widthO = width * factor;
    uint heightO = height * factor;
    auto *imgOut = (unsigned char *) calloc(widthO * heightO * 3, sizeof(unsigned char));
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

    for (int i = 0; i < widthO; i++)
    {
        for (int j = 0; j < heightO; ++j)
        {
            x = i / factor;
            y = j / factor;

            alpha = ((double) i / factor) - x;
            beta = ((double) j / factor) - y;

            createSquare(square, imgIn, x, y, width, height);

            for (int k = 0; k < channels; k++)
            {
                double p1 = cubicInterpolate(square[0][0 + k], square[1][0 + k], square[2][0 + k], square[3][0 + k], alpha);
                double p2 = cubicInterpolate(square[4][0 + k], square[5][0 + k], square[6][0 + k], square[7][0 + k], alpha);
                double p3 = cubicInterpolate(square[8][0 + k], square[9][0 + k], square[10][0 + k], square[11][0 + k], alpha);
                double p4 = cubicInterpolate(square[12][0 + k], square[13][0 + k], square[14 + k][0 + k], square[15][0 + k], alpha);
                double p = cubicInterpolate(p1, p2, p3, p4, beta);

                if (p > 255)
                    p = 255;
                else if (p < 0)
                    p = 0;

                imgOut[(i + j * widthO) * 3 + k] = (int) p;

            }
        }
    }


    *oWidth = widthO;
    *oHeight = heightO;

    return imgOut;
}
unsigned char *upscaleOmpBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *hWidth, uint *oHeight, int nThread)
{
    return nullptr;
}
unsigned char *upscaleCudaBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{
    //init
    uint wf, hf;

    //host
    wf = width * factor;
    hf = height * factor;
    uint iSize = width * height * 3;
    mlock(imgIn, iSize * sizeof(unsigned char));
    uint iSizeO = wf * hf * 3;
    auto h_imgOut = (unsigned char *) malloc(iSizeO * sizeof(unsigned char));
    if (h_imgOut == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Errore nell'allocazione della memoria\n");
        munlock(imgIn, iSize * sizeof(unsigned char));
        return nullptr;
    }

    //device
    unsigned char *d_imgIn;
    unsigned char *d_imgOut;
    cudaMalloc(&d_imgIn, iSize * sizeof(unsigned char));
    cudaMalloc(&d_imgOut, iSizeO * sizeof(unsigned char));

    //copy
    cudaMemcpy(d_imgIn, imgIn, iSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //upscale
    dim3 gridSize = {(wf + 7) / 8, (hf + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    bicubicUpscaleCUDA<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, height, factor, gridSize, blockSize);


    //copy back
    cudaMemcpy(h_imgOut, d_imgOut, iSizeO * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //free
    munlock(imgIn, iSize * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    *oWidth = wf;
    *oHeight = hf;

    return h_imgOut;
}
