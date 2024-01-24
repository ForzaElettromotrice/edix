#include "functions.cuh"

int parseBlurArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int radius = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (imgIn == nullptr || pathOut == nullptr || radius == 0)
    {
        handle_error("Invalid arguments for blur function.\n");
    }

    //TODO: leggere le immagini in base alla loro estensione
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
        oImg = blurSerial(img, width, height, radius, &oWidth, &oHeight);

    } else if (strcmp(tpp, "OMP") == 0)
    {
        img = loadPPM(imgIn, &width, &height);
        oImg = blurOmp(img, width, height, radius, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        img = loadPPM(imgIn, &width, &height);
        oImg = blurCuda(img, width, height, radius, &oWidth, &oHeight);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for blur function.\n");
    }

    if (oImg != nullptr)
    {
        //TODO: scrivere le immagini in base alla loro estensione
        writePPM(pathOut, oImg, oWidth, oHeight, "P6");
        free(oImg);
    }
    free(img);
    free(tpp);

    return 0;
}


unsigned char *blurSerial(const unsigned char *imgIn, uint width, uint height, int radius, uint *oWidth, uint *oHeight)
{
    uint totalPixels = height * width;

    unsigned char *blurImage;

    blurImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * CHANNELS);
    if (blurImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore nell'allocare memoria\n");
        return nullptr;
    }

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            uint red = 0;
            uint green = 0;
            uint blue = 0;

            int num = 0;
            int curr_i;
            int curr_j;

            for (int m = -radius; m <= radius; m++)
            {
                for (int n = -radius; n <= radius; n++)
                {

                    curr_i = i + m;
                    curr_j = j + n;
                    if ((curr_i < 0) || (curr_i > width - 1) || (curr_j < 0) || (curr_j > height - 1)) continue;

                    red += imgIn[(3 * (curr_i + curr_j * width))];
                    green += imgIn[(3 * (curr_i + curr_j * width)) + 1];
                    blue += imgIn[(3 * (curr_i + curr_j * width)) + 2];

                    num++;
                }
            }
            red /= num;
            green /= num;
            blue /= num;

            blurImage[3 * (i + j * width)] = red;
            blurImage[3 * (i + j * width) + 1] = green;
            blurImage[3 * (i + j * width) + 2] = blue;
        }
    }
    *oWidth = width;
    *oHeight = height;

    return blurImage;
}
unsigned char *blurOmp(const unsigned char *imgIn, uint width, uint height, int radius, uint *oWidth, uint *oHeight, int nThread)
{
    uint totalPixels = height * width;

    unsigned char *blurImage;

    blurImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * CHANNELS);
    if (blurImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore nell'allocare memoria\n");
        return nullptr;
    }
//TODO: collapse
//TODO: schedule
//TODO: numero di thread ottimale
#pragma omp parallel for num_threads(nThread) default(none) shared(width, height, radius, imgIn, blurImage)
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; i++)
        {
            if (j > height)
                continue;
            uint red = 0;
            uint green = 0;
            uint blue = 0;

            int num = 0;
            int curr_i;
            int curr_j;

            for (int m = -radius; m <= radius; m++)
            {
                for (int n = -radius; n <= radius; n++)
                {

                    curr_i = i + m;
                    curr_j = j + n;
                    if ((curr_i < 0) || (curr_i > width - 1) || (curr_j < 0) || (curr_j > height - 1)) continue;

                    red += imgIn[(3 * (curr_i + curr_j * width))];
                    green += imgIn[(3 * (curr_i + curr_j * width)) + 1];
                    blue += imgIn[(3 * (curr_i + curr_j * width)) + 2];

                    num++;
                }
            }
            red /= num;
            green /= num;
            blue /= num;

            blurImage[3 * (i + j * width)] = red;
            blurImage[3 * (i + j * width) + 1] = green;
            blurImage[3 * (i + j * width) + 2] = blue;
        }
    }

    *oWidth = width;
    *oHeight = height;

    return blurImage;
}
unsigned char *blurCuda(const unsigned char *imgIn, uint width, uint height, int radius, uint *oWidth, uint *oHeight)
{
    return nullptr;
}
