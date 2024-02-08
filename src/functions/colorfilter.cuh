#ifndef EDIX_COLORFILTER_CUH
#define EDIX_COLORFILTER_CUH

#include <iostream>
#include "../utils.hpp"
#include <sys/mman.h>


__global__ void colorFilter(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, uint channels, int r, int g, int b, uint squareTolerance);

#endif //EDIX_COLORFILTER_CUH