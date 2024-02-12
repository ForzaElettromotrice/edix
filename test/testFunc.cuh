#ifndef EDIX_TEST_HPP
#define EDIX_TEST_HPP

#include <iostream>
#include <chrono>
#include <omp.h>
#include "../src/functions/parser.hpp"
#include "../src/utils.hpp"


typedef struct
{
    long time;
    long threads;
    double speedup;
    double efficiency;
} performance_t;

void testBlur(char *message, unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp);
void testColorFilter(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp);
void testComposition(char *message, const unsigned char *img1, uint width1, uint height1, uint channels1, const unsigned char *img2, uint width2, uint height2, uint channels2, performance_t *bestOmp);
void testOverlap(char *message, const unsigned char *img1, uint width1, uint height1, uint channels1, const unsigned char *img2, uint width2, uint height2, uint channels2, performance_t *bestOmp);
void testGrayscale(char *message, const unsigned char *img, uint width, uint height, performance_t *bestOmp);
void testUpscaleBilinear(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp);
void testUpscaleBicubic(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp);
void testDownscaleBilinear(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp);
void testDownscaleBicubic(char *message, const unsigned char *img, uint width, uint height, uint channels, performance_t *bestOmp);


double efficiency(long timeSer, long timePar, int nThreads);
double speedup(long timeSer, long timePar);
void saveResult(char *message);

void testAccuracy(const unsigned char *bigImage, const unsigned char *smallImage, const unsigned char *grayImage, uint width1, uint height1, uint width2, uint height2, uint width3, uint height3);
void testPerformance(const unsigned char *imgIn1, const unsigned char *imgIn2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2);

#endif //EDIX_TEST_HPP

