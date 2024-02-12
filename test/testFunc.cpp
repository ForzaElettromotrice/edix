#include "testFunc.hpp"

// Restituisce la percentuale di miglioramento del tempo t2 rispetto t1
double percentage(long time1, long time2)
{
    return (((double) (time1 - time2) / (double) time1) * 100.0);
}

void print_perc(double perc)
{
    char *str = perc > 0.0 ? (char *) "Miglioramento" : (char *) "Peggioramento";
    printf("%s del " BOLD RED "%.2f\n" RESET, str, perc);
}

long print_times(auto time1, auto time2)
{
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
    printf("Execution time: %ldms\n", time.count());
    return time.count();
}


void testOld(const unsigned char *imgIn1, const unsigned char *imgIn2, const uint width1, const uint height1, const uint width2, const uint height2, const uint channels1, const uint channels2)
{
    uint outW, outH;
    long timeSer, timePar;

    printf(BOLD YELLOW "Esecuzione seriale\n" RESET);
    auto startSer = std::chrono::high_resolution_clock::now();
    /* Scegli la funzione da chiamare Serial */

    //free(blurSerial(imgIn1, *width1, *height1, 5, &outW, &outH));
    //free(grayscaleSerial(imgIn1, *width1, *height1, &outW, &outH));
    //free(colorFilterSerial(imgIn1, *width1, *height1, 0, 0, 255, 0, &outW, &outH));
    //free(overlapSerial(imgIn1, imgIn2, *width1, *height1, *width2, *height2, 100, 200, &outW, &outH));
    //free(compositionSerial(imgIn1, imgIn2, *width1, *height1, *width2, *height2, UP, &outW, &outH));
//    free(upscaleSerialBilinear(imgIn1, *width1, *height1, 3, 1, &outW, &outH));
    //free(upscaleSerialBicubic(imgIn1, *width1, *height1, 1, &outW, &outH));
    //free(downscaleSerialBilinear(imgIn1, *width1, *height1, 1, &outW, &outH));

    auto endSer = std::chrono::high_resolution_clock::now();
    timeSer = print_times(startSer, endSer);
    printf(BOLD YELLOW "%s" RESET, "\nEsecuzioni parallele\n");
    for (int i = 2; i <= 20; i++)
    {
        timePar = 0L;
        for (int j = 0; j < 100; j++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            /* Scegli la funzione parallela (OMP, CUDA) */

            //free(blurOmp(imgIn1, *width1, *height1, 5, &outW, &outH, i));
            //free(grayscaleOmp(imgIn1, *width1, *height1, &outW, &outH, i));
            //free(colorFilterOmp(imgIn1, *width1, *height1, 0, 0, 255, 0, &outW, &outH, i));
            //free(overlapOmp(imgIn1, imgIn2, *width1, *height1, *width2, *height2, 20, 20, &outW, &outH, i));
            //free(compositionOmp(imgIn1, imgIn2, *width1, *height1, *width2, *height2, UP, &outW, &outH, i));
//            free(upscaleOmpBilinear(imgIn1, *width1, *height1, 3, 1, &outW, &outH, i));
            //free(upscaleOmpBicubic(imgIn1, *width1, *height1, 1, &outW, &outH, i));
            //free(downscaleOmpBilinear(imgIn1, *width1, *height1, 1, &outW, &outH, i));

            auto end = std::chrono::high_resolution_clock::now();
            timePar += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
        double efficiency = ((double) timeSer / (i * (double) timePar / 100)) * 100,
                speedup = (double) timeSer / ((double) timePar / 100);

        printf(BOLD "threads: " RESET BLUE "%2d " RESET BOLD "time: " RESET GREEN "%.2f " RESET, i, (double) timePar / 100);
        printf(BOLD "S: " RESET RED "%.2f " RESET, speedup);
        printf(BOLD "E: " RESET BLUE "%.2f%s\n" RESET, efficiency, "%");
    }
}


void testAccuracy(const unsigned char *bigImage, const unsigned char *smallImage, const unsigned char *grayImage, uint width1, uint height1, uint width2, uint height2, uint width3, uint height3)
{
    uint oWidth;
    uint oHeight;
    unsigned char *imgOut;

    auto *grayTo3 = (unsigned char *) malloc(width3 * height3 * sizeof(unsigned char));
    memcpy(grayTo3, grayImage, width3 * height3 * sizeof(unsigned char));
    grayTo3 = from1To3Channels(grayTo3, width3, height3);

    const unsigned char *images[] = {bigImage, smallImage, grayImage};
    const char *name[] = {"Big", "Small", "Gray"};
    uint imagesW[] = {width1, width2, width3};
    uint imagesH[] = {height1, height2, height3};
    uint imagesC[] = {3, 3, 1};
    char *title = (char *) malloc(256 * sizeof(char));

    const char *side[] = {"Up", "Down", "Left", "Right"};
    const char *scale[] = {"Downscale", "Upscale"};

    D_Print("Testing blur...\n");
    for (int k = 0; k < 3; ++k)
    {
        imgOut = blurSerial(images[k], imagesW[k], imagesH[k], imagesC[k], 10, &oWidth, &oHeight);
        sprintf(title, "out/BlurSerial%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = blurOmp(images[k], imagesW[k], imagesH[k], imagesC[k], 10, &oWidth, &oHeight, 8, 8);
        sprintf(title, "out/BlurOmp%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = blurCuda(images[k], imagesW[k], imagesH[k], imagesC[k], 10, &oWidth, &oHeight, true);
        sprintf(title, "out/BlurCuda%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
    }

    D_Print("Testing colorFilter...\n");
    for (int k = 0; k < 3; ++k)
    {
        imgOut = colorFilterSerial(images[k], imagesW[k], imagesH[k], imagesC[k], 255, 0, 0, 0, &oWidth, &oHeight);
        sprintf(title, "out/ColorFilterSerial%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = colorFilterOmp(images[k], imagesW[k], imagesH[k], imagesC[k], 255, 0, 0, 0, &oWidth, &oHeight, 16);
        sprintf(title, "out/ColorFilterOmp%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = colorFilterCuda(images[k], imagesW[k], imagesH[k], imagesC[k], 255, 0, 0, 0, &oWidth, &oHeight);
        sprintf(title, "out/ColorFilterCuda%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
    }


    images[2] = grayTo3;
    imagesC[2] = 3;

    D_Print("Testing composition...\n");
    for (int k = 0; k < 2; ++k)
    {
        for (int i = 0; i < 4; ++i)
        {
            imgOut = compositionSerial(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], i, &oWidth, &oHeight);
            sprintf(title, "out/CompositionSerial%s%s.ppm", name[k], side[i]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = compositionOmp(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], i, &oWidth, &oHeight, 16);
            sprintf(title, "out/CompositionOmp%s%s.ppm", name[k], side[i]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = compositionOmpAlternative(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], i, &oWidth, &oHeight, 16);
            sprintf(title, "out/CompositionOmpAlternative%s%s.ppm", name[k], side[i]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = compositionCuda(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], i, &oWidth, &oHeight);
            sprintf(title, "out/CompositionCuda%s%s.ppm", name[k], side[i]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
        }
    }

    images[2] = grayImage;
    imagesC[2] = 1;

    D_Print("Testing grayscale...\n");
    for (int k = 0; k < 2; ++k)
    {
        imgOut = grayscaleSerial(images[k], imagesW[k], imagesH[k], &oWidth, &oHeight);
        sprintf(title, "out/GrayScaleSerial%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, 1);
        free(imgOut);
        imgOut = grayscaleOmp(images[k], imagesW[k], imagesH[k], &oWidth, &oHeight, 16);
        sprintf(title, "out/GrayScaleOmp%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, 1);
        free(imgOut);
        imgOut = grayscaleCuda(images[k], imagesW[k], imagesH[k], &oWidth, &oHeight);
        sprintf(title, "out/GrayScaleCuda%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, 1);
        free(imgOut);
    }

    images[2] = grayTo3;
    imagesC[2] = 3;

    D_Print("Testing overlap...\n");
    for (int k = 0; k < 2; ++k)
    {
        imgOut = overlapSerial(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], 0, 0, &oWidth, &oHeight);
        sprintf(title, "out/OverlapSerial%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = overlapOmp(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], 0, 0, &oWidth, &oHeight, 16);
        sprintf(title, "out/OverlapOmp%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = overlapCuda(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], 0, 0, &oWidth, &oHeight);
        sprintf(title, "out/OverlapCuda%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
    }

    images[2] = grayImage;
    imagesC[2] = 1;

    D_Print("Testing upscale and downscale...\n");
    for (int k = 0; k < 3; ++k)
    {
        for (int i = 0; i < 1; ++i)
        {
            imgOut = scaleSerialBilinear(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight);
            sprintf(title, "out/%sSerialBilinear%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleSerialBicubic(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight);
            sprintf(title, "out/%sSerialBicubic%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleOmpBilinear(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight, 16);
            sprintf(title, "out/%sOmpBilinear%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleOmpBicubic(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight, 16);
            sprintf(title, "out/%sOmpBicubic%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleCudaBilinear(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight, true);
            sprintf(title, "out/%sCudaBilinear%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleCudaBicubic(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight, true);
            sprintf(title, "out/%sCudaBicubic%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
        }
    }

    free(grayTo3);
    free(title);
}
void testPerformance(const unsigned char *imgIn1, const unsigned char *imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2);