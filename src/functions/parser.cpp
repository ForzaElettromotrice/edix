//
// Created by f3m on 08/02/24.
//

#include "parser.hpp"
#include <cstring>

int parseBlurArgs(char *args)
{

    char *imgIn = strtok(args, " ");
    if (imgIn == nullptr)
    {
        E_Print("usage " BOLD "funx blur IN OUT RADIUS\n" RESET);
        return 1;
    }
    char *pathOut = strtok(nullptr, " ");
    if (pathOut == nullptr)
    {
        E_Print("usage " BOLD "funx blur IN OUT RADIUS\n" RESET);
        return 1;
    }
    int radius;
    const char *nptr = strtok(nullptr, " ");
    if (nptr == nullptr)
    {
        E_Print("usage " BOLD "funx blur IN OUT RADIUS\n" RESET);
        return 1;
    } else
    {
        radius = (int) strtol(nptr, nullptr, 10);
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
        oImg = blurSerial(img, width, height, channels, radius, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = blurOmp(img, width, height, channels, radius, &oWidth, &oHeight, 10, 10);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = blurCuda(img, width, height, channels, radius, &oWidth, &oHeight, true);
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
    if (imgIn == nullptr)
    {
        E_Print("usage " BOLD "funx colorfilter IN OUT R G B TOLERANCE\n" RESET);
        return 1;
    }
    char *pathOut = strtok(nullptr, " ");
    if (pathOut == nullptr)
    {
        E_Print("usage " BOLD "funx colorfilter IN OUT R G B TOLERANCE\n" RESET);
        return 1;
    }
    char *arg_r = strtok(nullptr, " ");
    if (arg_r == nullptr)
    {
        E_Print("usage " BOLD "funx colorfilter IN OUT R G B TOLERANCE\n" RESET);
        return 1;
    }
    if (isNotNumber(arg_r))
    {
        E_Print("usage " BOLD "Il valore di R deve essere numerico\n" RESET);
        return 1;
    }
    int r = (int) strtol(arg_r, nullptr, 10);
    if (r > 255 || r < 0)
    {
        E_Print("usage " BOLD "Il valore di R deve essere compreso tra 0 e 255\n" RESET);
        return 1;
    }
    char *arg_g = strtok(nullptr, " ");
    if (arg_g == nullptr)
    {
        E_Print("usage " BOLD "funx colorfilter IN OUT R G B TOLERANCE\n" RESET);
        return 1;
    }
    if (isNotNumber(arg_g))
    {
        E_Print("usage " BOLD "Il valore di G deve essere numerico\n" RESET);
        return 1;
    }
    int g = (int) strtol(arg_g, nullptr, 10);
    if (g > 255 || g < 0)
    {
        E_Print("usage " BOLD "Il valore di G deve essere compreso tra 0 e 255\n" RESET);
        return 1;
    }
    char *arg_b = strtok(nullptr, " ");
    if (arg_b == nullptr)
    {
        E_Print("usage " BOLD "funx colorfilter IN OUT R G B TOLERANCE\n" RESET);
        return 1;
    }
    if (isNotNumber(arg_b))
    {
        E_Print("usage " BOLD "Il valore di B deve essere numerico\n" RESET);
        return 1;
    }
    int b = (int) strtol(arg_b, nullptr, 10);
    if (b > 255 || b < 0)
    {
        E_Print("usage " BOLD "Il valore di B deve essere compreso tra 0 e 255\n" RESET);
        return 1;
    }
    char *arg_tollerance = strtok(nullptr, " ");
    if (arg_tollerance == nullptr)
    {
        E_Print("usage " BOLD "funx colorfilter IN OUT R G B TOLERANCE\n" RESET);
        return 1;
    }
    if (isNotNumber(arg_tollerance))
    {
        E_Print("usage " BOLD "Il valore di TOLERANCE deve essere numerico\n" RESET);
        return 1;
    }
    uint tollerance = (uint) strtoul(arg_tollerance, nullptr, 10);

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
    if (pathIn == nullptr)
    {
        E_Print("usage " BOLD "funx upscale IN OUT FACTOR\n" RESET);
        return 1;
    }
    char *pathOut = strtok(nullptr, " ");
    if (pathOut == nullptr)
    {
        E_Print("usage " BOLD "funx upscale IN OUT FACTOR\n" RESET);
        return 1;
    }
    char *arg_factor = strtok(nullptr, " ");
    if (arg_factor == nullptr)
    {
        E_Print("usage " BOLD "funx upscale IN OUT FACTOR\n" RESET);
        return 1;
    }
    if (isNotNumber(arg_factor))
    {
        E_Print("usage " BOLD "Il valore di FACTOR deve essere numerico\n" RESET);
        return 1;
    }
    int factor = (int) strtol(arg_factor, nullptr, 10);
    if (factor < 1)
    {
        E_Print("usage " BOLD "Il fattore di upscaling deve essere maggiore di 0\n" RESET);
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
    if (pathIn == nullptr)
    {
        E_Print("usage " BOLD "funx downscale IN OUT FACTOR\n" RESET);
        return 1;
    }
    char *pathOut = strtok(nullptr, " ");
    if (pathOut == nullptr)
    {
        E_Print("usage " BOLD "funx downscale IN OUT FACTOR\n" RESET);
        return 1;
    }
    char *arg_factor = strtok(nullptr, " ");
    if (arg_factor == nullptr)
    {
        E_Print("usage " BOLD "funx downscale IN OUT FACTOR\n" RESET);
        return 1;
    }
    if (isNotNumber(arg_factor))
    {
        E_Print("usage " BOLD "Il valore di FACTOR deve essere numerico\n" RESET);
        return 1;
    }
    int factor = (int) strtol(arg_factor, nullptr, 10);
    if (factor < 1)
    {
        E_Print("usage " BOLD "Il fattore di downscaling deve essere maggiore di 0\n" RESET);
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
    char *path1 = strtok(args, " ");
    if (path1 == nullptr)
    {
        E_Print("usage " BOLD "funx overlap IN1 IN2 OUT X Y\n" RESET);
        return 1;
    }
    char *path2 = strtok(nullptr, " ");
    if (path2 == nullptr)
    {
        E_Print("usage " BOLD "funx overlap IN1 IN2 OUT X Y\n" RESET);
        return 1;
    }
    char *pathOut = strtok(nullptr, " ");
    if (pathOut == nullptr)
    {
        E_Print("usage " BOLD "funx overlap IN1 IN2 OUT X Y\n" RESET);
        return 1;
    }
    char *arg_x = strtok(nullptr, " ");
    if (isNotNumber(arg_x))
    {
        E_Print("usage " BOLD "Il valore di X deve essere numerico\n" RESET);
        return 1;
    }
    uint x = (uint) strtoul(arg_x, nullptr, 10);
    char *arg_y = strtok(nullptr, " ");
    if (isNotNumber(arg_y))
    {
        E_Print("usage " BOLD "Il valore di Y deve essere numerico\n" RESET);
        return 1;
    }
    uint y = (uint) strtoul(arg_y, nullptr, 10);

    char *tpp = getStrFromKey((char *) "TPP");
    uint width1;
    uint height1;
    uint channels1;
    uint width2;
    uint height2;
    uint channels2;
    unsigned char *imgIn1 = loadImage(path1, &width1, &height1, &channels1);
    unsigned char *imgIn2 = loadImage(path2, &width2, &height2, &channels2);
    if (channels1 != 3 && channels2 == 3)
    {
        imgIn1 = from1To3Channels(imgIn1, width1, height1);
        channels1 = 3;
    }
    if (channels2 != 3 && channels1 == 3)
    {
        imgIn2 = from1To3Channels(imgIn2, width2, height2);
        channels2 = 3;
    }


    uint oWidth;
    uint oHeight;
    uint oChannels = channels1;
    unsigned char *oImg;


    if (strcmp(tpp, "Serial") == 0)
        oImg = overlapSerial(imgIn1, imgIn2, width1, height1, channels1, width2, height2, channels2, x, y, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = overlapOmp(imgIn1, imgIn2, width1, height1, channels1, width2, height2, channels2, x, y, &oWidth, &oHeight, 4);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = overlapCuda(imgIn1, imgIn2, width1, height1, channels1, width2, height2, channels2, x, y, &oWidth, &oHeight);
    else
    {
        free(imgIn1);
        free(imgIn2);
        free(tpp);
        E_Print("Invalid TPP\n");
        return 1;
    }

    if (oImg != nullptr)
    {
        writeImage(pathOut, oImg, oWidth, oHeight, oChannels);
        free(oImg);
    }

    free(imgIn1);
    free(imgIn2);
    free(tpp);
    return 0;
}
int parseCompositionArgs(char *args)
{
    char *path1 = strtok(args, " ");
    if (path1 == nullptr)
    {
        E_Print("usage " BOLD "funx composition IN1 IN2 OUT SIDE\n" RESET);
        return 1;
    }
    char *path2 = strtok(nullptr, " ");
    if (path2 == nullptr)
    {
        E_Print("usage " BOLD "funx composition IN1 IN2 OUT SIDE\n" RESET);
        return 1;
    }
    char *pathOut = strtok(nullptr, " ");
    if (pathOut == nullptr)
    {
        E_Print("usage " BOLD "funx composition IN1 IN2 OUT SIDE\n" RESET);
        return 1;
    }
    char *arg_side = strtok(nullptr, " ");
    if (isNotNumber(arg_side))
    {
        E_Print("usage " BOLD "Il valore di SIDE deve essere numerico\n" RESET);
        return 1;
    }
    int side = (int) strtol(arg_side, nullptr, 10);
    if (side > 3)
    {
        E_Print("usage " BOLD "Il valore di SIDE deve essere compreso tra 0 e 3\n" RESET);
        return 1;
    }

    char *tpp = getStrFromKey((char *) "TPP");
    uint width1;
    uint height1;
    uint channels1;
    uint width2;
    uint height2;
    uint channels2;
    unsigned char *imgIn1 = loadPPM(path1, &width1, &height1, &channels1);
    unsigned char *imgIn2 = loadPPM(path2, &width2, &height2, &channels2);
    if (channels1 != 3 && channels2 == 3)
    {
        imgIn1 = from1To3Channels(imgIn1, width1, height1);
        channels1 = 3;
    }
    if (channels2 != 3 && channels1 == 3)
    {
        imgIn2 = from1To3Channels(imgIn2, width2, height2);
        channels2 = 3;
    }

    uint oWidth;
    uint oHeight;
    uint oChannels = channels1;
    unsigned char *oImg;


    if (strcmp(tpp, "Serial") == 0)
        oImg = compositionSerial(imgIn1, imgIn2, width1, height1, channels1, width2, height2, channels2, side, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = compositionOmp(imgIn1, imgIn2, width1, height1, channels1, width2, height2, channels2, side, &oWidth, &oHeight, 20);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = compositionCuda(imgIn1, imgIn2, width1, height1, channels1, width2, height2, channels2, side, &oWidth, &oHeight);
    else
    {
        free(imgIn1);
        free(imgIn2);
        free(tpp);
        E_Print("Invalid arguments for composition function.\n");
        return 1;
    }

    if (oImg != nullptr)
    {
        writeImage(pathOut, oImg, oWidth, oHeight, oChannels);
        free(oImg);
    }
    free(imgIn1);
    free(imgIn2);
    free(tpp);
    return 0;
}

int isNotNumber(char *str)
{
    for (int i = 0; i < strlen(str); i++)
        if (!isdigit(str[i]))
            return 1;
    return 0;
}

