#include "functions.cuh"

int parseGrayscaleArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");

    if (imgIn == nullptr || pathOut == nullptr)
    {
        handle_error("Errore nel parsing degli argomenti\n");
    }

    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(imgIn, &width, &height, &channels);
    if (channels != 3)
    {
        free(img);
        handle_error("Canali non validi per una scala di grigi!\n");
    }

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
        oImg = grayscaleSerial(img, width, height, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = grayscaleOmp(img, width, height, &oWidth, &oHeight, 4);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = grayscaleCuda(img, width, height, &oWidth, &oHeight);
    else
    {
        free(img);
        free(tpp);
        handle_error("Errore nel parsing degli argomenti\n");
    }
    if (oImg != nullptr)
    {
        writeImage(pathOut, oImg, oWidth, oHeight, 1);
        free(oImg);
    }
    free(img);
    free(tpp);
    return 0;
}

unsigned char *grayscaleSerial(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight)
{
    auto *img_out = (unsigned char *) malloc((width * height) * sizeof(unsigned char));

    if (img_out == nullptr)
    {
        fprintf(stderr, RED
        "Error: "
        RESET
        "Errore nell'allocazione della memoria\n");
        return nullptr;
    }


    int r;
    int g;
    int b;
    int i = 0;
    int grayValue;


    for (int y = 0; y < height; y += 1)
    {
        for (int x = 0; x < width; x += 1)
        {
            r = imgIn[((y * width) + x) * 3];
            g = imgIn[((y * width) + x) * 3 + 1];
            b = imgIn[((y * width) + x) * 3 + 2];

            grayValue = (r + g + b) / 3;

            img_out[i++] = grayValue;
        }
    }

    *oWidth = width;
    *oHeight = height;

    return img_out;
}
unsigned char *grayscaleOmp(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight, int nThread)
{
    auto *img_out = (unsigned char *) malloc((width * height) * sizeof(unsigned char));

    if (img_out == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Errore nell'allocazione della memoria");
        return nullptr;
    }

    int r, g, b, grayValue;

    //TODO: schedule e controllo su num_threads
#pragma omp parallel for num_threads(nThread) default(none) private(r, g, b, grayValue) shared(img_out, imgIn, width, height)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            r = imgIn[((y * width) + x) * 3];
            g = imgIn[((y * width) + x) * 3 + 1];
            b = imgIn[((y * width) + x) * 3 + 2];
            grayValue = (r + g + b) / 3;

            img_out[y * width + x] = grayValue;
        }
    }

    *oWidth = width;
    *oHeight = height;

    return img_out;
}
unsigned char *grayscaleCuda(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight)
{
    return nullptr;
}

