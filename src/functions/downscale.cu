//
// Created by f3m on 19/01/24.
//


#include "downscale.cuh"

int parseDownscaleArgs(char *args)
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
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = downscaleSerialBilinear(img, width, height, channels, factor, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = downscaleSerialBicubic(img, width, height, channels, factor, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = downscaleOmpBilinear(img, width, height, channels, factor, &oWidth, &oHeight, 4);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = downscaleOmpBicubic(img, width, height, channels, factor, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = downscaleCudaBilinear(img, width, height, channels, factor, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = downscaleCudaBicubic(img, width, height, channels, factor, &oWidth, &oHeight);
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

unsigned char *downscaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{

    uint widthO = width / factor;
    uint heightO = height / factor;
    auto *imgOut = (unsigned char *) malloc((widthO * heightO * 3) * sizeof(unsigned char));

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
            x = i * factor;
            y = j * factor;
            alpha = 0.5;
            beta = 0.5;


            for (int k = 0; k < channels; ++k)
            {
                p00 = imgIn[(x + y * width) * channels + k];
                p01 = x + 1 >= width ? p00 : imgIn[(x + 1 + y * width) * channels + k];
                p10 = y + 1 >= height ? p00 : imgIn[(x + (y + 1) * width) * channels + k];
                p11 = x + 1 >= width || y + 1 >= height ? p00 : imgIn[(x + 1 + (y + 1) * width) * channels + k];


                imgOut[(i + j * widthO) * channels + k] = bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
            }
        }
    }


    *oWidth = widthO;
    *oHeight = heightO;

    return imgOut;
}
unsigned char *downscaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread)
{
    return nullptr;
}
unsigned char *downscaleCudaBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{
    return nullptr;
}

unsigned char *downscaleSerialBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{
    uint widthO = width / factor;
    uint heightO = height / factor;
    auto *imgOut = (unsigned char *) calloc(widthO * heightO * channels, sizeof(unsigned char));
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
            x = i * factor;
            y = j * factor;

//            alpha = ((double) i * factor) - x;
//            beta = ((double) j * factor) - y;

            alpha = 0.5;
            beta = 0.5;

            //TODO: i pixel mancanti devono essere la copia dell'originale
            createSquareD(square, imgIn, x, y, width, height, channels);

            for (int k = 0; k < channels; k++)
            {
                double p1 = cubicInterpolate(square[0][k], square[1][k], square[2][k], square[3][k], alpha);
                double p2 = cubicInterpolate(square[4][k], square[5][k], square[6][k], square[7][k], alpha);
                double p3 = cubicInterpolate(square[8][k], square[9][k], square[10][k], square[11][k], alpha);
                double p4 = cubicInterpolate(square[12][k], square[13][k], square[14 + k][k], square[15][k], alpha);
                double p = cubicInterpolate(p1, p2, p3, p4, beta);

                if (p > 255)
                    p = 255;
                else if (p < 0)
                    p = 0;

                imgOut[(i + j * widthO) * channels + k] = (int) p;
            }
        }
    }


    *oWidth = widthO;
    *oHeight = heightO;

    return imgOut;
}
unsigned char *downscaleOmpBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread)
{
    return nullptr;
}
unsigned char *downscaleCudaBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{
    //init
    uint wf, hf;

    //host
    wf = width / factor;
    hf = height / factor;
    uint iSize = width * height * 3;
    uint iSizeO = wf * hf * 3;
    auto h_imgOut = (unsigned char *) malloc(iSizeO * sizeof(unsigned char));
    if (h_imgOut == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Errore nell'allocazione della memoria\n");
        munlock(imgIn, iSize * sizeof(unsigned char));
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
    bicubicDownscaleCUDA<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, height, factor, channels);

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

void createSquareD(unsigned char square[16][3], const unsigned char *img, int x, int y, uint width, uint height, uint channels)
{
    for (int i = -1; i < 3; ++i)
        for (int j = -1; j < 3; ++j)
        {
            if (x - i < 0 || y - j < 0 || x + i >= width || y + j >= height)
                continue;
            for (int k = 0; k < channels; ++k)
                square[(i + 1) + (j + 1) * 4][k] = img[channels * (x + i + (y + j) * width) + k];;
        }
}

__device__ void createSquareDEVICED(unsigned char square[16][3], const unsigned char *img, int x, int y, uint width, uint height, uint channels)
{
    for (int i = -1; i < 3; ++i)
    {
        for (int j = -1; j < 3; ++j)
        {
            if (x - i < 0 || y - j < 0 || x + i >= width || y + j >= height)
                continue;
            for (int k = 0; k < channels; ++k)
                square[(i + 1) + (j + 1) * 4][k] = img[channels * (x + i + (y + j) * width) + k];
        }
    }

}
__device__ double cubicInterpolateDEVICED(double A, double B, double C, double D, double t)
{
    double a = -A / 2.0f + (3.0f * B) / 2.0f - (3.0f * C) / 2.0f + D / 2.0f;
    double b = A - (5.0f * B) / 2.0f + 2.0f * C - D / 2.0f;
    double c = -A / 2.0f + C / 2.0f;
    double d = B;
    return a * t * t * t + b * t * t + c * t + d;
}

__global__ void bilinearDownscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, int factor)
{
    int x = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    uint widthO = width / factor;
    uint heightO = height / factor;

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
        i = x * factor;
        j = y * factor;
        alpha = ((double) x * factor) - i;
        beta = ((double) y * factor) - j;

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
__global__ void bicubicDownscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, int factor, uint channels)
{
    int x = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    uint widthO = width / factor;
    uint heightO = height / factor;

    uint idx = x + y * widthO;

    int i;
    int j;
    double alpha;
    double beta;
    unsigned char square[16][3];
    int pixel[3];

    if (idx < widthO * heightO * channels)
    {
        i = x * factor;
        j = y * factor;

        alpha = 0.5;
        beta = 0.5;


        createSquareDEVICED(square, imgIn, i, j, width, height, channels);

        for (int k = 0; k < channels; k++)
        {
            double p1 = cubicInterpolateDEVICED(square[0][k], square[1][k], square[2][k], square[3][k], alpha);
            double p2 = cubicInterpolateDEVICED(square[4][k], square[5][k], square[6][k], square[7][k], alpha);
            double p3 = cubicInterpolateDEVICED(square[8][k], square[9][k], square[10][k], square[11][k], alpha);
            double p4 = cubicInterpolateDEVICED(square[12][k], square[13][k], square[14 + k][k], square[15][k], alpha);
            double p = cubicInterpolateDEVICED(p1, p2, p3, p4, beta);

            if (p > 255)
                p = 255;
            else if (p < 0)
                p = 0;

            pixel[k] = (int) p;

        }

        imgOut[idx * channels] = pixel[0];
        imgOut[(idx * channels) + 1] = pixel[1];
        imgOut[(idx * channels) + 2] = pixel[2];
    }
}