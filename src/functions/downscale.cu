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
            alpha = ((double) i * factor) - x + 0.5;
            beta = ((double) j * factor) - y + 0.5;


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
    return nullptr;
}
unsigned char *downscaleOmpBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread)
{
    return nullptr;
}
unsigned char *downscaleCudaBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{
    return nullptr;
}