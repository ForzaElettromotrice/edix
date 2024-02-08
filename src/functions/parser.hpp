//
// Created by f3m on 08/02/24.
//

#ifndef EDIX_PARSER_HPP
#define EDIX_PARSER_HPP

#include <iostream>
#include <cstdlib>
#include <cstring>
#include "imgutils.hpp"
#include "../dbutils/rdutils.hpp"

int parseBlurArgs(char *args);
int parseGrayscaleArgs(char *args);
int parseColorFilterArgs(char *args);
int parseUpscaleArgs(char *args);
int parseDownscaleArgs(char *args);
int parseOverlapArgs(char *args);
int parseCompositionArgs(char *args);

unsigned char *blurSerial(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight);
unsigned char *blurOmp(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight, int nThread);
unsigned char *blurCuda(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight);

unsigned char *grayscaleSerial(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight);
unsigned char *grayscaleOmp(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight, int nThread);
unsigned char *grayscaleCuda(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight);

unsigned char *colorFilterSerial(const unsigned char *imgIn, uint width, uint height, uint channels, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight);
unsigned char *colorFilterOmp(const unsigned char *imgIn, uint width, uint height, uint channels, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight, int nThread);
unsigned char *colorFilterCuda(const unsigned char *imgIn, uint width, uint height, uint channels, uint r, uint g, uint b, uint tolerance, uint *oWidth, uint *oHeight);

unsigned char *scaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight);
unsigned char *scaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight, int nThreads);
unsigned char *scaleCudaBilinear(const unsigned char *h_imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight, bool useShared);


unsigned char *scaleSerialBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight);
unsigned char *scaleOmpBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight, int nThreads);
unsigned char *scaleCudaBicubic(const unsigned char *h_imgIn, uint width, uint height, uint channels, int factor, bool upscale, uint *oWidth, uint *oHeight, bool useShared);

unsigned char *overlapSerial(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight);
unsigned char *overlapOmp(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight, int nThread);
unsigned char *overlapCuda(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight);

unsigned char *compositionSerial(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight);
unsigned char *compositionOmp(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight, int nThread);
unsigned char *compositionCuda(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, int side, uint *oWidth, uint *oHeight);


#endif //EDIX_PARSER_HPP
