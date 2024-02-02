//
// Created by f3m on 19/01/24.
//

#include "functions.cuh"

int parseColorFilterArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    //TODO: controllare che i valori siano compresi tra 0 e 255 e che non siano state inserite stringhe
    uint r = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);
    uint g = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);
    uint b = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);
    uint tollerance = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);

    if (imgIn == nullptr || pathOut == nullptr || r > 255 || g > 255 || b > 255)
    {
        handle_error("Invalid arguments for color filter function.\n");
    }

    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(imgIn, &width, &height, &channels);

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
        oImg = colorFilterSerial(img, width, height, channels, r, g, b, tollerance, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = colorFilterOmp(img, width, height, channels, r, g, b, tollerance, &oWidth, &oHeight, 4);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = colorFilterCuda(img, width, height, channels, r, g, b, tollerance, &oWidth, &oHeight);
    else
    {
        free(img);
        free(tpp);
        handle_error("Invalid arguments for color filter function.\n");
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

unsigned char *colorFilterSerial(const unsigned char *imgIn, uint width, uint height, uint channels, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight)
{
    //TODO: rifare tutta la funzione
    unsigned char *filteredImage;

    uint totalPixels = width * height;

    filteredImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * 3);
    if (filteredImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore durante l'allocazione di memoria");
        return nullptr;
    }

    for (int i = 0; i < 3 * width * height; i += 3)
    {
        int diffR;
        int diffG;
        int diffB;
        if (channels == 3)
        {
            diffR = imgIn[i] - (int) r;
            diffG = imgIn[i + 1] - (int) g;
            diffB = imgIn[i + 2] - (int) b;
        } else
        {
            diffR = imgIn[i] - (int) r;
            diffG = imgIn[i] - (int) g;
            diffB = imgIn[i] - (int) b;
        }

        // Calcola la distanza euclidea nel cubo RGB
        uint distance = diffR * diffR + diffG * diffG + diffB * diffB;

        // Applica la soglia di tolleranza per filtrare il colore desiderato
        if (channels == 3)
        {
            if (distance > tolerance * tolerance)
            {
                // Riduci la saturazione degli altri colori
                filteredImage[i] = (imgIn[i] + (int) r) / 2;
                filteredImage[i + 1] = (imgIn[i + 1] + (int) g) / 2;
                filteredImage[i + 2] = (imgIn[i + 2] + (int) b) / 2;
            } else
            {
                filteredImage[i] = imgIn[i];
                filteredImage[i + 1] = imgIn[i + 1];
                filteredImage[i + 2] = imgIn[i + 2];
            }
        } else
        {
            if (distance > tolerance * tolerance)
            {
                // Riduci la saturazione degli altri colori
                filteredImage[i] = (imgIn[i] + (int) r) / 2;
                filteredImage[i + 1] = (imgIn[i] + (int) g) / 2;
                filteredImage[i + 2] = (imgIn[i] + (int) b) / 2;
            } else
            {
                filteredImage[i] = imgIn[i];
                filteredImage[i + 1] = imgIn[i];
                filteredImage[i + 2] = imgIn[i];
            }
        }
    }

    *oWidth = width;
    *oHeight = height;

    return filteredImage;
}
unsigned char *colorFilterOmp(const unsigned char *imgIn, uint width, uint height, uint channels, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight, int nThread)
{
    uint diffR,
            diffG,
            diffB,
            distance,
            totalPixels = width * height;

    unsigned char *filteredImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * 3);

    if (filteredImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore durante l'allocazione di memoria");
        return nullptr;
    }

#pragma omp parallel for num_threads(nThread) \
    default(none) private(diffR, diffG, diffB, distance) shared(filteredImage, imgIn, width, height, r, g, b, tolerance) \
    collapse(2)
    // TODO: prova a vedere se si puo' incrementare l'efficienza
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint rPix = ((y * width) + x) * 3,
                    gPix = ((y * width) + x) * 3 + 1,
                    bPix = ((y * width) + x) * 3 + 2;
            diffR = imgIn[rPix] - r;
            diffG = imgIn[gPix] - g;
            diffB = imgIn[bPix] - b;

            distance = (diffR * diffR) + (diffG * diffG) + (diffB * diffB);

            if (distance > tolerance * tolerance)
            {
                filteredImage[rPix] = (imgIn[rPix] + r) / 2;
                filteredImage[gPix] = (imgIn[gPix] + g) / 2;
                filteredImage[bPix] = (imgIn[bPix] + b) / 2;
            } else
            {
                filteredImage[rPix] = imgIn[rPix];
                filteredImage[gPix] = imgIn[gPix];
                filteredImage[bPix] = imgIn[bPix];
            }
        }
    }
    *oWidth = width;
    *oHeight = height;
    return filteredImage;
}
unsigned char *colorFilterCuda(const unsigned char *imgIn, uint width, uint height, uint channels, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight)
{
    return nullptr;
}
