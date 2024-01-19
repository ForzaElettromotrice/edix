//
// Created by f3m on 19/01/24.
//

#include "functions.hu"

int parseColorFilterArgs(char *args)
{
    return 0;
}

int colorFilterSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, uint r, uint g, uint b, uint tolerance)
{
    unsigned char* filteredImage;
    uint targetColor[3] = {r,g,b};

    uint totalPixels= width*height;

	filteredImage = (unsigned char *)malloc(sizeof(unsigned char*) * totalPixels * CHANNELS);
    if (filteredImage == nullptr) {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore durante l'allocazione di memoria");
        return 1;
    }

    for (int i = 0; i <  3 * width * height ; i+=3) {

            int diffR = imgIn[i] - targetColor[0];
            int diffG = imgIn[i+1] - targetColor[1];
            int diffB = imgIn[i+2] - targetColor[2];

        // Calcola la distanza euclidea nel cubo RGB
            int distance = diffR * diffR + diffG * diffG + diffB * diffB;

        // Applica la soglia di tolleranza per filtrare il colore desiderato
            if (distance > tolerance * tolerance) {
            // Riduci la saturazione degli altri colori
                filteredImage[i] = (imgIn[i] + targetColor[0]) / 2;
                filteredImage[i+ 1] = (imgIn[i+ 1] + targetColor[1]) / 2;
                filteredImage[i+ 2] = (imgIn[i+ 2] + targetColor[2]) / 2;
            } else{
                filteredImage[i] = imgIn[i];
                filteredImage[i+1] = imgIn[i+1];
                filteredImage[i+2] = imgIn[i+2];
            }
    }

    writePPM(strcat(pathOut,".ppm"), filteredImage, width, height,"P6");
    free(filteredImage);

    return 0;
}
int colorFilterOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, uint r, uint g, uint b)
{
    return 0;
}
int colorFilterCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, uint r, uint g, uint b)
{
    return 0;
}
