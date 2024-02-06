//
// Created by f3m on 06/02/24.
//

#ifndef EDIX_SCALE_CUH
#define EDIX_SCALE_CUH

#include "../utils.hpp"
#include <sys/mman.h>


int h_bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta);
__device__ int d_bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta);

__global__ void scale(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint oWidth, uint oHeight, uint channels, int factor, bool upscale);
__global__ void scaleShared(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint oWidth, uint oHeight, uint channels, int factor);


#endif //EDIX_SCALE_CUH
