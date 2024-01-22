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


    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        blurSerial(img, pathOut, width, height, radius);
        free(img);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        blurOmp(img, pathOut, width, height, radius);
        free(img);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        blurCuda(img, pathOut, width, height, radius);
        free(img);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for blur function.\n");
    }
    free(tpp);

    return 0;
}


int blurSerial(const unsigned char *imgIn, char *pathOut, uint width, uint height, int radius)
{
    uint totalPixels = height * width;

    unsigned char *blurImage;

    blurImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * CHANNELS);
    if (blurImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore nell'allocare memoria\n");
        return 1;
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


    writePPM(pathOut, blurImage, width, height, "P6");
    free(blurImage);

    return 0;
}
int blurOmp(const unsigned char *imgIn, char *pathOut, uint width, uint height, int radius)
{
    uint totalPixels = height * width;

    unsigned char *blurImage;

    blurImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * CHANNELS);
    if (blurImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore nell'allocare memoria\n");
        return 1;
    }
//TODO: collapse
//TODO: schedule
//TODO: numero di thread ottimale
#pragma omp parallel for num_threads(omp_get_max_threads()) default(none) shared(width, height, radius, imgIn, blurImage)
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


    writePPM(pathOut, blurImage, width, height, "P6");
    free(blurImage);
    return 0;
}
int blurCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius)
{
    return 0;
}
