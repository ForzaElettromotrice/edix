
#ifndef EDIX_GRAYSCALE_HU
#define EDIX_GRAYSCALE_HU

#include <iostream>
#include <cstdlib>
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


//parsers
int parseBlurArgs(char *args);
int parseGrayscaleArgs(char *args);
int parseColorFilterArgs(char *args);
int parseUpscaleArgs(char *args);
int parseDownscaleArgs(char *args);
int parseOverlapArgs(char *args);
int parseCompositionArgs(char *args);


//funx
int blurSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius);
int blurOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius);
int blurCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius);

int grayscaleSerial(unsigned char *imgIn, char *pathOut, uint width, uint height);
int grayscaleOmp(unsigned char *imgIn, char *pathOut, uint width, uint height);
int grayscaleCuda(unsigned char *imgIn, char *pathOut, uint width, uint height);

int colorFilterSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, uint r, uint g, uint b);
int colorFilterOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, uint r, uint g, uint b);
int colorFilterCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, uint r, uint g, uint b);

int overlapSerial(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2,
                  uint height2, uint x,
                  uint y);
int
overlapOmp(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2,
           uint height2, uint x, uint y);
int
overlapCuda(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2,
            uint height2, uint x,
            uint y);

int compositionSerial(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2,
                      uint height2, int side);
int compositionOmp(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2,
                   uint height2, int side);
int compositionCuda(unsigned char *img1, unsigned char *img2, char *pathOut, uint width1, uint height1, uint width2,
                    uint height2, int side);

int upscaleSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor);
int upscaleOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor);
int upscaleCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor);

int downscaleSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor);
int downscaleOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor);
int downscaleCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int factor);


#endif  //EDIX_GRAYSCALE_HU
