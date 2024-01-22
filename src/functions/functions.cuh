
#ifndef EDIX_GRAYSCALE_HU
#define EDIX_GRAYSCALE_HU

#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <chrono>
#include "../dbutils/rdutils.hpp"
#include "../utils.hpp"

#define CHANNELS 3

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

// utils
unsigned char *loadPPM(const char *path, uint *width, uint *height);
void writePPM(const char *path, unsigned char *img, uint width, uint height, const char *format);
unsigned char *jpegDecode(const char *path, int *width, int *height);
unsigned char *pngDecode(const char *path, int *width, int *height);

//parsers
int parseBlurArgs(char *args);
int parseGrayscaleArgs(char *args);
int parseColorFilterArgs(char *args);
int parseUpscaleArgs(char *args);
int parseDownscaleArgs(char *args);
int parseOverlapArgs(char *args);
int parseCompositionArgs(char *args);


//funx

unsigned char *blurSerial(const unsigned char *imgIn, uint width, uint height, int radius, uint *oWidth, uint *oHeight);
unsigned char *blurOmp(const unsigned char *imgIn, uint width, uint height, int radius, uint *oWidth, uint *oHeight);
unsigned char *blurCuda(const unsigned char *imgIn, uint width, uint height, int radius, uint *oWidth, uint *oHeight);


unsigned char *grayscaleSerial(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight);
unsigned char *grayscaleOmp(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight);
unsigned char *grayscaleCuda(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight);


unsigned char *colorFilterSerial(const unsigned char *imgIn, uint width, uint height, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight);
unsigned char *colorFilterOmp(const unsigned char *imgIn, uint width, uint height, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight);
unsigned char *colorFilterCuda(const unsigned char *imgIn, uint width, uint height, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight);


unsigned char *overlapSerial(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint width2, uint height2, uint x, uint y, uint *oWidth, uint *oHeight);
unsigned char *overlapOmp(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint width2, uint height2, uint x, uint y, uint *oWidth, uint *oHeight);
unsigned char *overlapCuda(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint width2, uint height2, uint x, uint y, uint *oWidth, uint *oHeight);

unsigned char *compositionSerial(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint width2, uint height2, int side, uint *oWidth, uint *oHeight);
unsigned char *compositionOmp(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint width2, uint height2, int side, uint *oWidth, uint *oHeight);
unsigned char *compositionCuda(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint width2, uint height2, int side, uint *oWidth, uint *oHeight);

unsigned char *upscaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, int factor, uint *hWidth, uint *oHeight);
unsigned char *upscaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, int factor, uint *hWidth, uint *oHeight);
unsigned char *upscaleCudaBilinear(const unsigned char *imgIn, uint width, uint height, int factor, uint *hWidth, uint *oHeight);

unsigned char *downscaleSerial(const unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight);
unsigned char *downscaleOmp(const unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight);
unsigned char *downscaleCuda(const unsigned char *imgIn, uint width, uint height, int factor, uint *oWidth, uint *oHeight);


#endif  //EDIX_GRAYSCALE_HU
