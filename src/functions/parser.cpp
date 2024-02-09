//
// Created by f3m on 08/02/24.
//

#include "parser.hpp"
#include <cstring>

int parseBlurArgs(char *args)
{
   
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    if (pathOut == nullptr){
        E_Print("usage " BOLD "funx blur IN OUT RADIUS\n" RESET);
        return 1;
    }
    int radius;
    const char *nptr = strtok(nullptr, " ");
    if (nptr == nullptr){
        E_Print("usage " BOLD "funx blur IN OUT RADIUS\n" RESET);
        return 1;
    }else{
        radius = (int) strtol(nptr, nullptr, 10);
    }


    //TODO: leggere le immagini in base alla loro estensione
    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(imgIn, &width, &height, &channels);

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;


    if (strcmp(tpp, "Serial") == 0)
        oImg = blurSerial(img, width, height, channels, radius, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = blurOmp(img, width, height, channels, radius, &oWidth, &oHeight, 4);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = blurCuda(img, width, height, channels, radius, &oWidth, &oHeight);
    else
    {
        free(img);
        free(tpp);
        E_Print("Invalid arguments for blur function.\n");
        return 1;
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
int parseGrayscaleArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");

    if (imgIn == nullptr || pathOut == nullptr)
    {
        E_Print("usage " BOLD "funx grayscale IN OUT\n" RESET);
        return 1;
    }

    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(imgIn, &width, &height, &channels);
    if (channels != 3)
    {
        free(img);
        E_Print("Canali non validi per una scala di grigi!\n");
        return 1;
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
        E_Print("Errore nel parsing degli argomenti\n");
        return 1;
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
int parseColorFilterArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    //TODO: controllare che i valori siano compresi tra 0 e 255 e che non siano state inserite stringhe
    int r = (int) strtol(strtok(nullptr, " "), nullptr, 10);
    int g = (int) strtol(strtok(nullptr, " "), nullptr, 10);
    int b = (int) strtol(strtok(nullptr, " "), nullptr, 10);
    uint tollerance = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);

    if (imgIn == nullptr || pathOut == nullptr || r > 255 || g > 255 || b > 255)
    {
        E_Print("usage " BOLD "funx colorfilter IN OUT R G B TOLERANCE\n" RESET);
        return 1;
    }

    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(imgIn, &width, &height, &channels);
    if (channels == 1)
        img = from1To3Channels(img, width, height);

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
        E_Print("Invalid arguments for color filter function.\n");
        return 1;
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
int parseUpscaleArgs(char *args)
{
    char *pathIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int factor = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (pathIn == nullptr || pathOut == nullptr || factor == 0)
    {
        E_Print("usage " BOLD "funx upscale IN OUT FACTOR\n" RESET);
        return 1;
    }

    char *tpp = getStrFromKey((char *) "TPP");
    char *tup = getStrFromKey((char *) "TUP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(pathIn, &width, &height, &channels);

    uint oWidth = 0;
    uint oHeight = 0;
    unsigned char *imgOut;

    if (strcmp(tpp, "Serial") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = scaleSerialBilinear(img, width, height, channels, factor, true, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = scaleSerialBicubic(img, width, height, channels, factor, true, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = scaleOmpBilinear(img, width, height, channels, factor, true, &oWidth, &oHeight, 4);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = scaleOmpBicubic(img, width, height, channels, factor, true, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            imgOut = scaleCudaBilinear(img, width, height, channels, factor, true, &oWidth, &oHeight, true);
        else if (strcmp(tup, "Bicubic") == 0)
            imgOut = scaleCudaBicubic(img, width, height, channels, factor, true, &oWidth, &oHeight, true);
    } else
    {
        free(img);
        free(tpp);
        E_Print("Invalid TPP\n");
        return 1;
    }

    if (imgOut != nullptr)
    {
        writeImage(pathOut, imgOut, oWidth, oHeight, channels);
        free(imgOut);
    }
    free(img);
    free(tpp);
    return 0;
}
int parseDownscaleArgs(char *args)
{
    char *pathIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int factor = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (pathIn == nullptr || pathOut == nullptr || factor == 0)
    {
        E_Print("usage " BOLD "funx downscale IN OUT FACTOR\n" RESET);
        return 1;
    }

    char *tpp = getStrFromKey((char *) "TPP");
    char *tup = getStrFromKey((char *) "TUP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(pathIn, &width, &height, &channels);

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = scaleSerialBilinear(img, width, height, channels, factor, false, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = scaleSerialBicubic(img, width, height, channels, factor, false, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = scaleOmpBilinear(img, width, height, channels, factor, false, &oWidth, &oHeight, 4);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = scaleOmpBicubic(img, width, height, channels, factor, false, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = scaleCudaBilinear(img, width, height, channels, factor, false, &oWidth, &oHeight, false);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = scaleCudaBicubic(img, width, height, channels, factor, false, &oWidth, &oHeight, false);
    } else
    {
        free(tpp);
        E_Print("Invalid TPP\n");
        return 1;
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
        E_Print("usage " BOLD "funx overlap IN1 IN2 OUT SIDE\n" RESET);
        return 1;
    }

    char *tpp = getStrFromKey((char *) "TPP");
    uint width1;
    uint height1;
    uint channels1;
    uint width2;
    uint height2;
    uint channels2;
    unsigned char *img1_1 = loadImage(img1, &width1, &height1, &channels1);
    unsigned char *img2_1 = loadImage(img2, &width2, &height2, &channels2);

    uint oWidth;
    uint oHeight;
    uint oChannels = channels1 == 3 ? 3 : channels2;
    unsigned char *oImg;


    if (strcmp(tpp, "Serial") == 0)
        oImg = overlapSerial(img1_1, img2_1, width1, height1, channels1, width2, height2, channels2, x, y, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = overlapOmp(img1_1, img2_1, width1, height1, channels1, width2, height2, channels2, x, y, &oWidth, &oHeight, 4);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = overlapCuda(img1_1, img2_1, width1, height1, channels1, width2, height2, channels2, x, y, &oWidth, &oHeight);
    else
    {
        free(img1_1);
        free(img2_1);
        free(tpp);
        E_Print("Invalid TPP\n");
        return 1;
    }

    if (oImg != nullptr)
    {
        writeImage(pathOut, oImg, oWidth, oHeight, oChannels);
        free(oImg);
    }

    free(img1_1);
    free(img2_1);
    free(tpp);
    return 0;
}
int parseCompositionArgs(char *args)
{
    char *img1 = strtok(args, " ");
    char *img2 = strtok(nullptr, " ");
    char *pathOut = strtok(nullptr, " ");
    //TODO: check meglio (crasha se ci sono troppi pochi valori)
    int side = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (img1 == nullptr || img2 == nullptr || pathOut == nullptr)
    {
        E_Print("usage " BOLD "funx composition IN1 IN2 OUT SIDE\n" RESET);
        return 1;
    }

    char *tpp = getStrFromKey((char *) "TPP");
    uint width1;
    uint height1;
    uint channels1;
    uint width2;
    uint height2;
    uint channels2;
    unsigned char *img1_1 = loadPPM(img1, &width1, &height1, &channels1);
    unsigned char *img2_1 = loadPPM(img2, &width2, &height2, &channels2);

    uint oWidth;
    uint oHeight;
    uint oChannels = channels1 == 3 ? 3 : channels2;
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
        oImg = compositionSerial(img1_1, img2_1, width1, height1, channels1, width2, height2, channels2, side, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = compositionOmp(img1_1, img2_1, width1, height1, channels1, width2, height2, channels2, side, &oWidth, &oHeight, 3);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = compositionCuda(img1_1, img2_1, width1, height1, channels1, width2, height2, channels2, side, &oWidth, &oHeight);
    else
    {
        free(img1_1);
        free(img2_1);
        free(tpp);
        E_Print("Invalid arguments for composition function.\n");
        return 1;
    }

    if (oImg != nullptr)
    {
        writeImage(pathOut, oImg, oWidth, oHeight, oChannels);
        free(oImg);
    }
    free(img1_1);
    free(img2_1);
    free(tpp);
    return 0;
}

