//
// Created by f3m on 19/01/24.
//

#include "downscale.cuh"

int parseDownscaleArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int factor = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (imgIn == nullptr || pathOut == nullptr || factor == 0)
    {
        handle_error("Invalid arguments for downscale function.\n");
    }
    // TODO: leggere le immagini in base all'estenzione

    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;
    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        downscaleSerial(img, pathOut, width, height, factor);
        free(img);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        downscaleOmp(img, pathOut, width, height, factor);
        free(img);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        downscaleCuda(img, pathOut, width, height, factor);
        free(img);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for downscale function.\n");
    }

    free(tpp);
    return 0;
}

int downscaleSerial(const unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
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
            //TODO: capire bene come calcolare alpha e beta
            x = i * factor;
            y = j * factor;
            alpha = ((double) i * factor) - x + 0.5;
            beta = ((double) j * factor) - y + 0.5;


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
int downscaleOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}
int downscaleCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor)
{
    return 0;
}