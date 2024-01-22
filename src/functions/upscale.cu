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
    char *tup = getStrFromKey((char *) "TUP");
    uint width;
    uint height;
    unsigned char *img;

    uint oWidth;
    uint oHeight;
    unsigned char *imgOut;

    if (strcmp(tpp, "Serial") == 0)
    {
        img = loadPPM(img1, &width, &height);
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = upscaleSerialBilinear(img, width, height, factor, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = upscaleSerialBicubic(img, width, height, factor, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        img = loadPPM(img1, &width, &height);
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = upscaleOmpBilinear(img, width, height, factor, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = upscaleOmpBicubic(img, width, height, factor, &oWidth, &oHeight);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        img = loadPPM(img1, &width, &height);
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = upscaleCudaBilinear(img, width, height, factor, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = upscaleCudaBicubic(img, width, height, factor, &oWidth, &oHeight);
    } else
    {
        free(tpp);
        handle_error("Invalid TPP\n");
    }

    if (imgOut != nullptr)
    {
        //TODO: salvare nel formato giusto
        writePPM(pathOut, imgOut, oWidth, oHeight, "P6");
        free(imgOut);
    }
    free(img);
    free(tpp);
    return 0;
}

unsigned char *upscaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight)
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
unsigned char *upscaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight)
{
    return nullptr;
}
unsigned char *upscaleCudaBilinear(const unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight)
{
    return nullptr;
}

unsigned char *upscaleSerialBicubic(const unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight)
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
    int pixel[3];

    for (int i = 0; i < widthO; i++)
    {
        for (int j = 0; j < heightO; ++j)
        {
            x = i / factor;
            y = j / factor;

            alpha = ((double) i / factor) - x;
            beta = ((double) j / factor) - y;

            createSquare(square, imgIn, x, y, width, height);

            for (int k = 0; k < 3; k++)
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

                pixel[k] = (int) p;

            }

//            printf("%d %d %d\n", pixel[0], pixel[1], pixel[2]);
            imgOut[(i + j * widthO) * 3] = pixel[0];
            imgOut[(i + j * widthO) * 3 + 1] = pixel[1];
            imgOut[(i + j * widthO) * 3 + 2] = pixel[2];
        }
    }


    *oWidth = widthO;
    *oHeight = heightO;

    return imgOut;
}
unsigned char *upscaleOmpBicubic(const unsigned char *imgIn, uint width, uint height, int factor, uint *hWidth, uint *oHeight)
{
    return nullptr;
}
unsigned char *upscaleCudaBicubic(const unsigned char *imgIn, uint width, uint height, int factor, uint *hWidth, uint *oHeight)
{
    return nullptr;
}