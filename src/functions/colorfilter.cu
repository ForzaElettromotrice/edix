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
        oImg = colorFilterSerial(img, width, height, r, g, b, tollerance, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        img = loadPPM(imgIn, &width, &height);
        oImg = colorFilterOmp(img, width, height, r, g, b, tollerance, &oWidth, &oHeight);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        img = loadPPM(imgIn, &width, &height);
        oImg = colorFilterCuda(img, width, height, r, g, b, tollerance, &oWidth, &oHeight);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for color filter function.\n");
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

unsigned char *colorFilterSerial(const unsigned char *imgIn, uint width, uint height, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight)
{
    unsigned char *filteredImage;

    uint totalPixels = width * height;

    filteredImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * CHANNELS);
    if (filteredImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore durante l'allocazione di memoria");
        return nullptr;
    }

    for (int i = 0; i < 3 * width * height; i += 3)
    {

        int diffR = imgIn[i] - (int) r;
        int diffG = imgIn[i + 1] - (int) g;
        int diffB = imgIn[i + 2] - (int) b;

        // Calcola la distanza euclidea nel cubo RGB
        uint distance = diffR * diffR + diffG * diffG + diffB * diffB;

        // Applica la soglia di tolleranza per filtrare il colore desiderato
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
    }

    *oWidth = width;
    *oHeight = height;

    return filteredImage;
}
unsigned char *colorFilterOmp(const unsigned char *imgIn, uint width, uint height, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight)
{
    uint diffR,
         diffG,
         diffB,
         distance,
         totalPixels = width * height; 

    unsigned char *filteredImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * CHANNELS);

    if (filteredImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore durante l'allocazione di memoria");
        return nullptr;
    }   

    #pragma omp parallel for num_threads(4) \
    default(none) private(diffR, diffG, diffB, distance) shared(filteredImage, imgIn, width, height, r, g, b, tolerance) \
    collapse(2) 
    // TODO: prova a vedere se si puo' incrementare l'efficienza
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint rPix = ((y * width) + x) * 3,
                 gPix = ((y * width) + x) * 3 + 1, 
                 bPix = ((y * width) + x) * 3 + 2;
            diffR = imgIn[rPix] - r;
            diffG = imgIn[gPix] - g;
            diffB = imgIn[bPix] - b;

            distance = (diffR * diffR) + (diffG * diffG) + (diffB * diffB); 

            if (distance > tolerance * tolerance) {
                filteredImage[rPix] = (imgIn[rPix] + r) / 2;
                filteredImage[gPix] = (imgIn[gPix] + g) / 2;
                filteredImage[bPix] = (imgIn[bPix] + b) / 2;
            } else {
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
unsigned char *colorFilterCuda(const unsigned char *imgIn, uint width, uint height, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight)
{
    return nullptr;
}
