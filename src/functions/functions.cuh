
#ifndef EDIX_GRAYSCALE_HU
#define EDIX_GRAYSCALE_HU

#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <chrono>
#include "imgutils.hpp"
#include "../dbutils/rdutils.hpp"
#include "../utils.hpp"

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

// utils


//parsers
int parseBlurArgs(char *args);
int parseGrayscaleArgs(char *args);
int parseColorFilterArgs(char *args);
int parseUpscaleArgs(char *args);
int parseDownscaleArgs(char *args);
int parseOverlapArgs(char *args);
int parseCompositionArgs(char *args);


//funx

unsigned char *blurSerial(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight);
unsigned char *blurOmp(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight, int nThread);
unsigned char *blurCuda(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight);


unsigned char *grayscaleSerial(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight);
unsigned char *grayscaleOmp(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight, int nThread);
unsigned char *grayscaleCuda(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight);


unsigned char *colorFilterSerial(const unsigned char *imgIn, uint width, uint height, uint channels, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight);
unsigned char *colorFilterOmp(const unsigned char *imgIn, uint width, uint height, uint channels, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight, int nThread);
unsigned char *colorFilterCuda(const unsigned char *imgIn, uint width, uint height, uint channels, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight);


unsigned char *overlapSerial(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight);
unsigned char *overlapOmp(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight, int nThread);
unsigned char *overlapCuda(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight);

unsigned char *compositionSerial(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight);
unsigned char *compositionOmp(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight, int nThread);
unsigned char *compositionCuda(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight);

unsigned char *upscaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);
unsigned char *upscaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread);
unsigned char *upscaleCudaBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);

unsigned char *upscaleSerialBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);
unsigned char *upscaleOmpBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread);
unsigned char *upscaleCudaBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);

unsigned char *downscaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);
unsigned char *downscaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread);
unsigned char *downscaleCudaBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);

unsigned char *downscaleSerialBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);
unsigned char *downscaleOmpBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread);
unsigned char *downscaleCudaBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);

unsigned char *upscaleCudaBilinearShared(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);
unsigned char *upscaleCudaBicubicShared(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);

unsigned char *downscaleCudaBilinearShared(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight);
#endif  //EDIX_GRAYSCALE_HU
