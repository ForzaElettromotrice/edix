//
// Created by f3m on 19/01/24.
//

#include "colorfilter.cuh"


__global__ void colorFilterCUDA(unsigned char *imgIn,unsigned char *imgOut, uint width, uint height, uint r, uint g, uint b, uint tolerance){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int i = idx * 3;

    if (i < 3 * width * height)
    {
        int diffR = imgIn[i] - (int)r;
        int diffG = imgIn[i + 1] - (int)g;
        int diffB = imgIn[i + 2] - (int)b;

        uint distance = diffR * diffR + diffG * diffG + diffB * diffB;

        if (distance > tolerance * tolerance)
        {
            imgOut[i] = (imgIn[i] + (int)r) / 2;
            imgOut[i + 1] = (imgIn[i + 1] + (int)g) / 2;
            imgOut[i + 2] = (imgIn[i + 2] + (int)b) / 2;
        }
        else
        {
            imgOut[i] = imgIn[i];
            imgOut[i + 1] = imgIn[i + 1];
            imgOut[i + 2] = imgIn[i + 2];
        }

    }
}

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
    char format[3];

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
    {
        img = loadPPM(imgIn, &width, &height, format);
        oImg = colorFilterSerial(img, width, height, r, g, b, tollerance, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        img = loadPPM(imgIn, &width, &height, format);
        oImg = colorFilterOmp(img, width, height, r, g, b, tollerance, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        img = loadPPM(imgIn, &width, &height, format);
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

    filteredImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * 3);
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
unsigned char *colorFilterOmp(const unsigned char *imgIn, uint width, uint height, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight, int nThread)
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
unsigned char *colorFilterCuda(const unsigned char *imgIn, uint width, uint height, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight)
{
    unsigned char *h_imgOut;

    unsigned char *d_imgOut;
    unsigned char *d_imgIn;
    

    mlock(imgIn,width * height * 3 * sizeof(unsigned char));
    h_imgOut = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));
    if (h_imgOut == nullptr){
        fprintf(stderr, RED "Error: " RESET "Errore nell'allocazione della memoria\n");
        munlock(imgIn, width * height * 3 * sizeof(unsigned char));
        return nullptr;
    }
    

    cudaMalloc((void **)&d_imgIn, width * height * 3 * sizeof(unsigned char));
    cudaMemcpy(d_imgIn,imgIn,width * height * 3 * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_imgOut,width * height * 3 * sizeof(unsigned char));


    uint blockSize = 8 * 8 * 3;
    uint gridSize = (3 * width * height  + blockSize -1) /blockSize;
    colorFilterCUDA<<<gridSize,blockSize>>>(d_imgIn, d_imgOut ,width, height, r,  g, b, tolerance);
    

    cudaMemcpy(h_imgOut,d_imgOut,width * height * 3 * sizeof(unsigned char),cudaMemcpyDeviceToHost);
    
    munlock(imgIn, width * height * 3 * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    *oHeight = height;
    *oWidth = width;

    return h_imgOut;
}
