//
// Created by f3m on 19/01/24.
//

#include "upscale.cuh"

int bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta)
{
    return (int) ((1 - alpha) * (1 - beta) * p00 + (1 - alpha) * beta * p01 + alpha * (1 - beta) * p10 +
                  alpha * beta * p11);
}


int parseUpscaleArgs(char *args)
{
    char *img1 = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int factor = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (img1 == nullptr || pathOut == nullptr || factor == 0)
    {
        handle_error("Invalid arguments for upscale\n");
    }

    // TODO: leggere le immagini in base all'estensione
    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;

    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img = loadPPM(img1, &width, &height);
        upscaleSerial(img, pathOut, width, height, factor);
        free(img);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img = loadPPM(img1, &width, &height);
        upscaleOmp(img, pathOut, width, height, factor);
        free(img);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img = loadPPM(img1, &width, &height);
        upscaleCuda(img, pathOut, width, height, factor);
        free(img);
    } else
    {
        free(tpp);
        handle_error("Invalid TPP\n");
    }
    free(tpp);
    return 0;
}

int upscaleSerial(const unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{

    uint widthO = width * factor;
    uint heightO = height * factor;
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
            x = i / factor;
            y = j / factor;
            alpha = ((double) i / factor) - x;
            beta = ((double) j / factor) - y;

            for (int k = 0; k < 3; ++k)
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


    writePPM(pathOut, imgOut, widthO, heightO, "P6");
    free(imgOut);
    return 0;
}
int upscaleOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}
int upscaleCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}