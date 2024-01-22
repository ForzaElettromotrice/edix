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
    unsigned char *img;

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
    {
        img = loadPPM(imgIn, &width, &height);
        //TODO: vedere se bicubiva o bilineare
        oImg = downscaleSerial(img, width, height, factor, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        img = loadPPM(imgIn, &width, &height);
        //TODO: vedere se bicubiva o bilineare
        oImg = downscaleOmp(img, width, height, factor, &oWidth, &oHeight);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        img = loadPPM(imgIn, &width, &height);
        //TODO: vedere se bicubiva o bilineare
        oImg = downscaleCuda(img, width, height, factor, &oWidth, &oHeight);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for downscale function.\n");
    }

    if (oImg != nullptr)
    {
        writePPM(pathOut, oImg, oWidth, oHeight, "P6");
        free(oImg);
    }

    free(img);
    free(tpp);
    return 0;
}

unsigned char *downscaleSerial(const unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight)
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


    *oWidth = widthO;
    *oHeight = heightO;

    return imgOut;
}
unsigned char *downscaleOmp(unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight)
{
    return nullptr;
}
unsigned char *downscaleCuda(unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight)
{
    return nullptr;
}