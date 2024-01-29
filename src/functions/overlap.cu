//
// Created by f3m on 19/01/24.
//

#include "functions.cuh"

int parseOverlapArgs(char *args)
{
    char *img1 = strtok(args, " ");
    char *img2 = strtok(nullptr, " ");
    char *pathOut = strtok(nullptr, " ");
    //TODO: controllare se x e y sono stringhe o meno
    uint x = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);
    uint y = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);

    if (img1 == nullptr || img2 == nullptr || pathOut == nullptr)
    {
        handle_error("Invalid arguments for overlap\n");
    }
    // TODO: leggere le immagini in base all'estenzione

    char *tpp = getStrFromKey((char *) "TPP");
    uint width1;
    uint height1;
    uint width2;
    uint height2;
    unsigned char *img1_1;
    unsigned char *img2_1;

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;


    if (strcmp(tpp, "Serial") == 0)
    {
        img1_1 = loadPPM(img1, &width1, &height1);
        img2_1 = loadPPM(img2, &width2, &height2);
        oImg = overlapSerial(img1_1, img2_1, width1, height1, width2, height2, x, y, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        img1_1 = loadPPM(img1, &width1, &height1);
        img2_1 = loadPPM(img2, &width2, &height2);
        oImg = overlapOmp(img1_1, img2_1, width1, height1, width2, height2, x, y, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        img1_1 = loadPPM(img1, &width1, &height1);
        img2_1 = loadPPM(img2, &width2, &height2);
        oImg = overlapCuda(img1_1, img2_1, width1, height1, width2, height2, x, y, &oWidth, &oHeight);
    } else
    {
        free(tpp);
        handle_error("Invalid TPP\n");
    }

    if (oImg != nullptr)
    {
        writePPM(pathOut, oImg, oWidth, oHeight, "P6");
        free(oImg);
    }

    free(img1_1);
    free(img2_1);
    free(tpp);
    return 0;
}

unsigned char *overlapSerial(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint width2, uint height2, uint x, uint y, uint *oWidth, uint *oHeight)
{
    if (x + width2 > width1 || y + height2 > height1)
    {
        fprintf(stderr, RED "Error: " RESET "La secondo immagine è troppo grande per essere inserita li!\n");
        return nullptr;
    }

    auto *oImg = (unsigned char *) malloc(width1 * height1 * 3 * sizeof(unsigned char));

    memcpy(oImg, img1, width1 * height1 * 3 * sizeof(unsigned char));

    for (int i = 0; i < width2; i++)
        for (int j = 0; j < height2; j++)
        {
            oImg[3 * (x + i + (y + j) * width1)] = img2[3 * (i + j * width2)];
            oImg[3 * (x + i + (y + j) * width1) + 1] = img2[3 * (i + j * width2) + 1];
            oImg[3 * (x + i + (y + j) * width1) + 2] = img2[3 * (i + j * width2) + 2];
        }

    *oWidth = width1;
    *oHeight = height1;


    return oImg;
}
unsigned char *overlapOmp(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint width2, uint height2, uint x, uint y, uint *oWidth, uint *oHeight, int nThread)
{
    if (x + width2 > width1 || y + height2 > height1)
    {
        fprintf(stderr, RED "Error: " RESET "La secondo immagine è troppo grande per essere inserita li!\n");
        return nullptr;
    }
    auto *oImg = (unsigned char *) malloc(width1 * height1 * 3 * sizeof(unsigned char));

    memcpy(oImg, img1, width1 * height1 * 3 * sizeof(unsigned char));

    #pragma omp parallel for num_threads(nThread) \
            default(none) shared(img1, img2, width1, height1, width2, height2, x, y, oImg) \
            schedule(static)
    for (int i = 0; i < width2; i++) {
        for (int j = 0; j < height2; j++)
        {
            oImg[3 * (x + i + (y + j) * width1)] = img2[3 * (i + j * width2)];
            oImg[3 * (x + i + (y + j) * width1) + 1] = img2[3 * (i + j * width2) + 1];
            oImg[3 * (x + i + (y + j) * width1) + 2] = img2[3 * (i + j * width2) + 2];
        }
    }

    *oWidth = width1;
    *oHeight = height1;

    return oImg;
}
unsigned char *overlapCuda(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint width2, uint height2, uint x, uint y, uint *oWidth, uint *oHeight)
{
    return nullptr;
}