#include "testFunc.cuh"

void testBlur(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp)
{
    uint oWidth;
    uint oHeight;

    //testBlur
    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *imgOut = blurSerial(img, width, height, channels, 30, &oWidth, &oHeight);
    auto end = std::chrono::high_resolution_clock::now();
    long timeSer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    sprintf(message, "Serial time: %ld\n", timeSer);
    saveResult(message);
    free(imgOut);
    for (int i = 2; i < omp_get_max_threads() + 1; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        imgOut = blurOmp(img, width, height, channels, 30, &oWidth, &oHeight, i);
        end = std::chrono::high_resolution_clock::now();
        long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double s = speedup(timeSer, timePar);
        double e = efficiency(timeSer, timePar, i);
        sprintf(message, "Omp time: %ld\tnThread: %d\tSpeedup: %.2f\tEfficiency: %.2f\n", timePar, i, s, e);
        saveResult(message);
        if (bestOmp->time == 0 || bestOmp->time > timePar)
        {
            bestOmp->time = timePar;
            bestOmp->threads = i;
            bestOmp->speedup = s;
            bestOmp->efficiency = e;
        }
        free(imgOut);
    }
    sprintf(message, "Best omp time: %ld\tnThread: %ld\tSpeedup: %.2f\tEfficiency: %.2f\n", bestOmp->time, bestOmp->threads, bestOmp->speedup, bestOmp->efficiency);
    saveResult(message);
    start = std::chrono::high_resolution_clock::now();
    imgOut = blurCuda(img, width, height, channels, 30, &oWidth, &oHeight, false);
    end = std::chrono::high_resolution_clock::now();
    long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double s = speedup(timeSer, timePar);
    sprintf(message, "Cuda time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
    cudaDeviceReset();
    start = std::chrono::high_resolution_clock::now();
    imgOut = blurCuda(img, width, height, channels, 30, &oWidth, &oHeight, true);
    end = std::chrono::high_resolution_clock::now();
    timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    s = speedup(timeSer, timePar);
    sprintf(message, "Cuda shared time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
}
void testColorFilter(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp)
{
    uint oWidth;
    uint oHeight;

    //testBlur
    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *imgOut = colorFilterSerial(img, width, height, channels, 126, 58, 69, 50, &oWidth, &oHeight);
    auto end = std::chrono::high_resolution_clock::now();
    long timeSer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    sprintf(message, "Serial time: %ld\n", timeSer);
    saveResult(message);
    free(imgOut);
    for (int i = 2; i < omp_get_max_threads() + 1; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        imgOut = colorFilterOmp(img, width, height, channels, 126, 58, 69, 50, &oWidth, &oHeight, i);
        end = std::chrono::high_resolution_clock::now();
        long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double s = speedup(timeSer, timePar);
        double e = efficiency(timeSer, timePar, i);
        sprintf(message, "Omp time: %ld\tnThread: %d\tSpeedup: %.2f\tEfficiency: %.2f\n", timePar, i, s, e);
        saveResult(message);
        if (bestOmp->time == 0 || bestOmp->time > timePar)
        {
            bestOmp->time = timePar;
            bestOmp->threads = i;
            bestOmp->speedup = s;
            bestOmp->efficiency = e;
        }
        free(imgOut);
    }
    sprintf(message, "Best omp time: %ld\tnThread: %ld\tSpeedup: %.2f\tEfficiency: %.2f\n", bestOmp->time, bestOmp->threads, bestOmp->speedup, bestOmp->efficiency);
    saveResult(message);
    start = std::chrono::high_resolution_clock::now();
    imgOut = colorFilterCuda(img, width, height, channels, 126, 58, 69, 50, &oWidth, &oHeight);
    end = std::chrono::high_resolution_clock::now();
    long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double s = speedup(timeSer, timePar);
    sprintf(message, "Cuda time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
    cudaDeviceReset();
}
void testComposition(char *message, const unsigned char *img1, uint width1, uint height1, uint channels1, const unsigned char *img2, uint width2, uint height2, uint channels2, performance_t *bestOmp)
{
    uint oWidth;
    uint oHeight;

    //testBlur
    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *imgOut = compositionSerial(img1, img2, width1, height1, channels1, width2, height2, channels2, 2, &oWidth, &oHeight);
    auto end = std::chrono::high_resolution_clock::now();
    long timeSer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    sprintf(message, "Serial time: %ld\n", timeSer);
    saveResult(message);
    free(imgOut);
    for (int i = 2; i < omp_get_max_threads() + 1; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        imgOut = compositionOmp(img1, img2, width1, height1, channels1, width2, height2, channels2, 2, &oWidth, &oHeight, i);
        end = std::chrono::high_resolution_clock::now();
        long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double s = speedup(timeSer, timePar);
        double e = efficiency(timeSer, timePar, i);
        sprintf(message, "Omp time: %ld\tnThread: %d\tSpeedup: %.2f\tEfficiency: %.2f\n", timePar, i, s, e);
        saveResult(message);
        if (bestOmp->time == 0 || bestOmp->time > timePar)
        {
            bestOmp->time = timePar;
            bestOmp->threads = i;
            bestOmp->speedup = s;
            bestOmp->efficiency = e;
        }
        free(imgOut);
    }
    sprintf(message, "Best omp time: %ld\tnThread: %ld\tSpeedup: %.2f\tEfficiency: %.2f\n", bestOmp->time, bestOmp->threads, bestOmp->speedup, bestOmp->efficiency);
    saveResult(message);
    start = std::chrono::high_resolution_clock::now();
    imgOut = compositionCuda(img1, img2, width1, height1, channels1, width2, height2, channels2, 2, &oWidth, &oHeight);
    end = std::chrono::high_resolution_clock::now();
    long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double s = speedup(timeSer, timePar);
    sprintf(message, "Cuda time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
    cudaDeviceReset();
}
void testOverlap(char *message, const unsigned char *img1, uint width1, uint height1, uint channels1, const unsigned char *img2, uint width2, uint height2, uint channels2, performance_t *bestOmp)
{
    uint oWidth;
    uint oHeight;

    //testBlur
    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *imgOut = overlapSerial(img1, img2, width1, height1, channels1, width2, height2, channels2, 100, 100, &oWidth, &oHeight);
    auto end = std::chrono::high_resolution_clock::now();
    long timeSer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    sprintf(message, "Serial time: %ld\n", timeSer);
    saveResult(message);
    free(imgOut);
    for (int i = 2; i < omp_get_max_threads() + 1; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        imgOut = overlapOmp(img1, img2, width1, height1, channels1, width2, height2, channels2, 100, 100, &oWidth, &oHeight, i);
        end = std::chrono::high_resolution_clock::now();
        long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double s = speedup(timeSer, timePar);
        double e = efficiency(timeSer, timePar, i);
        sprintf(message, "Omp time: %ld\tnThread: %d\tSpeedup: %.2f\tEfficiency: %.2f\n", timePar, i, s, e);
        saveResult(message);
        if (bestOmp->time == 0 || bestOmp->time > timePar)
        {
            bestOmp->time = timePar;
            bestOmp->threads = i;
            bestOmp->speedup = s;
            bestOmp->efficiency = e;
        }
        free(imgOut);
    }
    sprintf(message, "Best omp time: %ld\tnThread: %ld\tSpeedup: %.2f\tEfficiency: %.2f\n", bestOmp->time, bestOmp->threads, bestOmp->speedup, bestOmp->efficiency);
    saveResult(message);
    start = std::chrono::high_resolution_clock::now();
    imgOut = overlapCuda(img1, img2, width1, height1, channels1, width2, height2, channels2, 100, 100, &oWidth, &oHeight);
    end = std::chrono::high_resolution_clock::now();
    long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double s = speedup(timeSer, timePar);
    sprintf(message, "Cuda time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
    cudaDeviceReset();
}
void testGrayscale(char *message, const unsigned char *img, uint width, uint height, performance_t *bestOmp)
{
    uint oWidth;
    uint oHeight;

    //testBlur
    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *imgOut = grayscaleSerial(img, width, height, &oWidth, &oHeight);
    auto end = std::chrono::high_resolution_clock::now();
    long timeSer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    sprintf(message, "Serial time: %ld\n", timeSer);
    saveResult(message);
    free(imgOut);
    for (int i = 2; i < omp_get_max_threads() + 1; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        imgOut = grayscaleOmp(img, width, height, &oWidth, &oHeight, i);
        end = std::chrono::high_resolution_clock::now();
        long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double s = speedup(timeSer, timePar);
        double e = efficiency(timeSer, timePar, i);
        sprintf(message, "Omp time: %ld\tnThread: %d\tSpeedup: %.2f\tEfficiency: %.2f\n", timePar, i, s, e);
        saveResult(message);
        if (bestOmp->time == 0 || bestOmp->time > timePar)
        {
            bestOmp->time = timePar;
            bestOmp->threads = i;
            bestOmp->speedup = s;
            bestOmp->efficiency = e;
        }
        free(imgOut);
    }
    sprintf(message, "Best omp time: %ld\tnThread: %ld\tSpeedup: %.2f\tEfficiency: %.2f\n", bestOmp->time, bestOmp->threads, bestOmp->speedup, bestOmp->efficiency);
    saveResult(message);
    start = std::chrono::high_resolution_clock::now();
    imgOut = grayscaleCuda(img, width, height, &oWidth, &oHeight);
    end = std::chrono::high_resolution_clock::now();
    long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double s = speedup(timeSer, timePar);
    sprintf(message, "Cuda time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
    cudaDeviceReset();
}
void testUpscaleBilinear(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp)
{
    uint oWidth;
    uint oHeight;

    //testBlur
    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *imgOut = scaleSerialBilinear(img, width, height, channels, 6, true, &oWidth, &oHeight);
    auto end = std::chrono::high_resolution_clock::now();
    long timeSer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    sprintf(message, "Serial time: %ld\n", timeSer);
    saveResult(message);
    free(imgOut);
    for (int i = 2; i < omp_get_max_threads() + 1; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        imgOut = scaleOmpBilinear(img, width, height, channels, 6, true, &oWidth, &oHeight, i);
        end = std::chrono::high_resolution_clock::now();
        long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double s = speedup(timeSer, timePar);
        double e = efficiency(timeSer, timePar, i);
        sprintf(message, "Omp time: %ld\tnThread: %d\tSpeedup: %.2f\tEfficiency: %.2f\n", timePar, i, s, e);
        saveResult(message);
        if (bestOmp->time == 0 || bestOmp->time > timePar)
        {
            bestOmp->time = timePar;
            bestOmp->threads = i;
            bestOmp->speedup = s;
            bestOmp->efficiency = e;
        }
        free(imgOut);
    }
    sprintf(message, "Best omp time: %ld\tnThread: %ld\tSpeedup: %.2f\tEfficiency: %.2f\n", bestOmp->time, bestOmp->threads, bestOmp->speedup, bestOmp->efficiency);
    saveResult(message);
    start = std::chrono::high_resolution_clock::now();
    imgOut = scaleCudaBilinear(img, width, height, channels, 6, true, &oWidth, &oHeight, false);
    end = std::chrono::high_resolution_clock::now();
    long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double s = speedup(timeSer, timePar);
    sprintf(message, "Cuda time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
    cudaDeviceReset();
    start = std::chrono::high_resolution_clock::now();
    imgOut = scaleCudaBilinear(img, width, height, channels, 6, true, &oWidth, &oHeight, true);
    end = std::chrono::high_resolution_clock::now();
    timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    s = speedup(timeSer, timePar);
    sprintf(message, "Cuda shared time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
}
void testUpscaleBicubic(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp)
{
    uint oWidth;
    uint oHeight;

    //testBlur
    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *imgOut = scaleSerialBicubic(img, width, height, channels, 6, true, &oWidth, &oHeight);
    auto end = std::chrono::high_resolution_clock::now();
    long timeSer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    sprintf(message, "Serial time: %ld\n", timeSer);
    saveResult(message);
    free(imgOut);
    for (int i = 2; i < omp_get_max_threads() + 1; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        imgOut = scaleOmpBicubic(img, width, height, channels, 6, true, &oWidth, &oHeight, i);
        end = std::chrono::high_resolution_clock::now();
        long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double s = speedup(timeSer, timePar);
        double e = efficiency(timeSer, timePar, i);
        sprintf(message, "Omp time: %ld\tnThread: %d\tSpeedup: %.2f\tEfficiency: %.2f\n", timePar, i, s, e);
        saveResult(message);
        if (bestOmp->time == 0 || bestOmp->time > timePar)
        {
            bestOmp->time = timePar;
            bestOmp->threads = i;
            bestOmp->speedup = s;
            bestOmp->efficiency = e;
        }
        free(imgOut);
    }
    sprintf(message, "Best omp time: %ld\tnThread: %ld\tSpeedup: %.2f\tEfficiency: %.2f\n", bestOmp->time, bestOmp->threads, bestOmp->speedup, bestOmp->efficiency);
    saveResult(message);
    start = std::chrono::high_resolution_clock::now();
    imgOut = scaleCudaBicubic(img, width, height, channels, 6, true, &oWidth, &oHeight, false);
    end = std::chrono::high_resolution_clock::now();
    long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double s = speedup(timeSer, timePar);
    sprintf(message, "Cuda time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
    cudaDeviceReset();
    start = std::chrono::high_resolution_clock::now();
    imgOut = scaleCudaBicubic(img, width, height, channels, 6, true, &oWidth, &oHeight, true);
    end = std::chrono::high_resolution_clock::now();
    timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    s = speedup(timeSer, timePar);
    sprintf(message, "Cuda shared time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
}
void testDownscaleBilinear(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp)
{
    uint oWidth;
    uint oHeight;

    //testBlur
    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *imgOut = scaleSerialBilinear(img, width, height, channels, 6, false, &oWidth, &oHeight);
    auto end = std::chrono::high_resolution_clock::now();
    long timeSer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    sprintf(message, "Serial time: %ld\n", timeSer);
    saveResult(message);
    free(imgOut);
    for (int i = 2; i < omp_get_max_threads() + 1; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        imgOut = scaleOmpBilinear(img, width, height, channels, 6, false, &oWidth, &oHeight, i);
        end = std::chrono::high_resolution_clock::now();
        long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double s = speedup(timeSer, timePar);
        double e = efficiency(timeSer, timePar, i);
        sprintf(message, "Omp time: %ld\tnThread: %d\tSpeedup: %.2f\tEfficiency: %.2f\n", timePar, i, s, e);
        saveResult(message);
        if (bestOmp->time == 0 || bestOmp->time > timePar)
        {
            bestOmp->time = timePar;
            bestOmp->threads = i;
            bestOmp->speedup = s;
            bestOmp->efficiency = e;
        }
        free(imgOut);
    }
    sprintf(message, "Best omp time: %ld\tnThread: %ld\tSpeedup: %.2f\tEfficiency: %.2f\n", bestOmp->time, bestOmp->threads, bestOmp->speedup, bestOmp->efficiency);
    saveResult(message);
    start = std::chrono::high_resolution_clock::now();
    imgOut = scaleCudaBilinear(img, width, height, channels, 6, false, &oWidth, &oHeight, false);
    end = std::chrono::high_resolution_clock::now();
    long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double s = speedup(timeSer, timePar);
    sprintf(message, "Cuda time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
    cudaDeviceReset();
    start = std::chrono::high_resolution_clock::now();
    imgOut = scaleCudaBilinear(img, width, height, channels, 6, false, &oWidth, &oHeight, true);
    end = std::chrono::high_resolution_clock::now();
    timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    s = speedup(timeSer, timePar);
    sprintf(message, "Cuda shared time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
}
void testDownscaleBicubic(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp)
{
    uint oWidth;
    uint oHeight;

    //testBlur
    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *imgOut = scaleSerialBicubic(img, width, height, channels, 6, false, &oWidth, &oHeight);
    auto end = std::chrono::high_resolution_clock::now();
    long timeSer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    sprintf(message, "Serial time: %ld\n", timeSer);
    saveResult(message);
    free(imgOut);
    for (int i = 2; i < omp_get_max_threads() + 1; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        imgOut = scaleOmpBicubic(img, width, height, channels, 6, false, &oWidth, &oHeight, i);
        end = std::chrono::high_resolution_clock::now();
        long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double s = speedup(timeSer, timePar);
        double e = efficiency(timeSer, timePar, i);
        sprintf(message, "Omp time: %ld\tnThread: %d\tSpeedup: %.2f\tEfficiency: %.2f\n", timePar, i, s, e);
        saveResult(message);
        if (bestOmp->time == 0 || bestOmp->time > timePar)
        {
            bestOmp->time = timePar;
            bestOmp->threads = i;
            bestOmp->speedup = s;
            bestOmp->efficiency = e;
        }
        free(imgOut);
    }
    sprintf(message, "Best omp time: %ld\tnThread: %ld\tSpeedup: %.2f\tEfficiency: %.2f\n", bestOmp->time, bestOmp->threads, bestOmp->speedup, bestOmp->efficiency);
    saveResult(message);
    start = std::chrono::high_resolution_clock::now();
    imgOut = scaleCudaBicubic(img, width, height, channels, 6, false, &oWidth, &oHeight, false);
    end = std::chrono::high_resolution_clock::now();
    long timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double s = speedup(timeSer, timePar);
    sprintf(message, "Cuda time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
    cudaDeviceReset();
    start = std::chrono::high_resolution_clock::now();
    imgOut = scaleCudaBicubic(img, width, height, channels, 6, false, &oWidth, &oHeight, true);
    end = std::chrono::high_resolution_clock::now();
    timePar = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    s = speedup(timeSer, timePar);
    sprintf(message, "Cuda shared time: %ld\tSpeedup: %.2f\n", timePar, s);
    saveResult(message);
    free(imgOut);
}


double efficiency(long timeSer, long timePar, int nThreads)
{
    return (double) timeSer / (nThreads * (double) timePar) * 100;
}
double speedup(long timeSer, long timePar)
{
    return (double) timeSer / ((double) timePar);
}
void saveResult(char *message)
{
    FILE *file = fopen("performance/performance.txt", "a");
    if (file == nullptr)
    {
        E_Print("Impossibile aprire il file!\n");
        return;
    }

    fwrite(message, sizeof(char), strlen(message), file);
    fclose(file);

    printf("%s", message);
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

    D_Print("Testing testBlur...\n");
    for (int k = 0; k < 3; ++k)
    {
        imgOut = blurSerial(images[k], imagesW[k], imagesH[k], imagesC[k], 10, &oWidth, &oHeight);
        sprintf(title, "accuracy/BlurSerial%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = blurOmp(images[k], imagesW[k], imagesH[k], imagesC[k], 10, &oWidth, &oHeight, 20);
        sprintf(title, "accuracy/BlurOmp%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = blurCuda(images[k], imagesW[k], imagesH[k], imagesC[k], 10, &oWidth, &oHeight, true);
        sprintf(title, "accuracy/BlurCuda%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
    }

    D_Print("Testing colorFilter...\n");
    for (int k = 0; k < 3; ++k)
    {
        imgOut = colorFilterSerial(images[k], imagesW[k], imagesH[k], imagesC[k], 255, 0, 0, 0, &oWidth, &oHeight);
        sprintf(title, "accuracy/ColorFilterSerial%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = colorFilterOmp(images[k], imagesW[k], imagesH[k], imagesC[k], 255, 0, 0, 0, &oWidth, &oHeight, 16);
        sprintf(title, "accuracy/ColorFilterOmp%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = colorFilterCuda(images[k], imagesW[k], imagesH[k], imagesC[k], 255, 0, 0, 0, &oWidth, &oHeight);
        sprintf(title, "accuracy/ColorFilterCuda%s.ppm", name[k]);
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
            sprintf(title, "accuracy/CompositionSerial%s%s.ppm", name[k], side[i]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = compositionOmp(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], i, &oWidth, &oHeight, 16);
            sprintf(title, "accuracy/CompositionOmp%s%s.ppm", name[k], side[i]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = compositionOmpAlternative(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], i, &oWidth, &oHeight, 16);
            sprintf(title, "accuracy/CompositionOmpAlternative%s%s.ppm", name[k], side[i]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = compositionCuda(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], i, &oWidth, &oHeight);
            sprintf(title, "accuracy/CompositionCuda%s%s.ppm", name[k], side[i]);
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
        sprintf(title, "accuracy/GrayScaleSerial%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, 1);
        free(imgOut);
        imgOut = grayscaleOmp(images[k], imagesW[k], imagesH[k], &oWidth, &oHeight, 16);
        sprintf(title, "accuracy/GrayScaleOmp%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, 1);
        free(imgOut);
        imgOut = grayscaleCuda(images[k], imagesW[k], imagesH[k], &oWidth, &oHeight);
        sprintf(title, "accuracy/GrayScaleCuda%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, 1);
        free(imgOut);
    }

    images[2] = grayTo3;
    imagesC[2] = 3;

    D_Print("Testing overlap...\n");
    for (int k = 0; k < 2; ++k)
    {
        imgOut = overlapSerial(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], 0, 0, &oWidth, &oHeight);
        sprintf(title, "accuracy/OverlapSerial%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = overlapOmp(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], 0, 0, &oWidth, &oHeight, 16);
        sprintf(title, "accuracy/OverlapOmp%s.ppm", name[k]);
        writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
        free(imgOut);
        imgOut = overlapCuda(images[k], images[k + 1], imagesW[k], imagesH[k], imagesC[k], imagesW[k + 1], imagesH[k + 1], imagesC[k + 1], 0, 0, &oWidth, &oHeight);
        sprintf(title, "accuracy/OverlapCuda%s.ppm", name[k]);
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
            sprintf(title, "accuracy/%sSerialBilinear%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleSerialBicubic(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight);
            sprintf(title, "accuracy/%sSerialBicubic%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleOmpBilinear(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight, 16);
            sprintf(title, "accuracy/%sOmpBilinear%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleOmpBicubic(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight, 16);
            sprintf(title, "accuracy/%sOmpBicubic%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleCudaBilinear(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight, true);
            sprintf(title, "accuracy/%sCudaBilinear%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
            imgOut = scaleCudaBicubic(images[k], imagesW[k], imagesH[k], imagesC[k], 2, i, &oWidth, &oHeight, true);
            sprintf(title, "accuracy/%sCudaBicubic%s.ppm", scale[i], name[k]);
            writeImage(title, imgOut, oWidth, oHeight, imagesC[k]);
            free(imgOut);
        }
    }

    free(grayTo3);
    free(title);
}
void testPerformance(const unsigned char *imgIn1, const unsigned char *imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2)
{

    char *message = (char *) malloc(256 * sizeof(char));
    auto *bestOmp = (performance_t *) calloc(1, sizeof(performance_t));

    sprintf(message, "Img1 = %dx%d\tChannels = %d\nImg2 = %dx%d\tChannels = %d\n", width1, height1, channels1, width2, height2, channels2);
    saveResult(message);

    sprintf(message, "Testing Blur: 30\n");
    saveResult(message);
    testBlur(message, imgIn1, width1, height1, channels1, bestOmp);
    memset(bestOmp, 0, sizeof(performance_t));

    sprintf(message, "Testing Colorfilter: r = 126\tg = 58\tb = 69\ttollerance = 50\n");
    saveResult(message);
    testColorFilter(message, imgIn1, width1, height1, channels1, bestOmp);
    memset(bestOmp, 0, sizeof(performance_t));

    sprintf(message, "Testing Composition: LEFT\n");
    saveResult(message);
    testComposition(message, imgIn1, width1, height1, channels1, imgIn2, width2, height2, channels2, bestOmp);
    memset(bestOmp, 0, sizeof(performance_t));

    sprintf(message, "Testing Grayscale\n");
    saveResult(message);
    testGrayscale(message, imgIn1, width1, height1, bestOmp);
    memset(bestOmp, 0, sizeof(performance_t));

    sprintf(message, "Testing Overlap: x = 100\ty = 100\n");
    saveResult(message);
    testOverlap(message, imgIn1, width1, height1, channels1, imgIn2, width2, height2, channels2, bestOmp);
    memset(bestOmp, 0, sizeof(performance_t));

    sprintf(message, "Testing Upscale Bilinear: 6\n");
    saveResult(message);
    testUpscaleBilinear(message, imgIn1, width1, height1, channels1, bestOmp);
    memset(bestOmp, 0, sizeof(performance_t));

    sprintf(message, "Testing Upscale Bicubic: 6\n");
    saveResult(message);
    testUpscaleBicubic(message, imgIn1, width1, height1, channels1, bestOmp);
    memset(bestOmp, 0, sizeof(performance_t));

    sprintf(message, "Testing Downscale Bilinear: 6\n");
    saveResult(message);
    testDownscaleBilinear(message, imgIn1, width1, height1, channels1, bestOmp);
    memset(bestOmp, 0, sizeof(performance_t));

    sprintf(message, "Testing Downscale Bicubic: 6\n");
    saveResult(message);
    testDownscaleBicubic(message, imgIn1, width1, height1, channels1, bestOmp);
    memset(bestOmp, 0, sizeof(performance_t));


    free(bestOmp);
    free(message);
}
