//
// Created by f3m on 06/02/24.
//

#ifndef EDIX_SCALE_CUH
#define EDIX_SCALE_CUH

#include <iostream>
#include "../utils.hpp"
#include <sys/mman.h>


__host__ __device__ int bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta);
__host__ __device__ void createSquare(unsigned char square[16][3], const unsigned char *img, int x, int y, uint width, uint height, uint channels);
__host__ __device__ double bicubicInterpolation(double A, double B, double C, double D, double t);

__global__ void scaleBilinear(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint oWidth, uint oHeight, uint channels, int factor, bool upscale);
__global__ void scaleBilinearShared(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint oWidth, uint oHeight, uint channels, int factor);

__global__ void scaleBicubic(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint oWidth, uint oHeight, uint channels, int factor, bool upscale);
__global__ void scaleBicubicShared(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint oWidth, uint oHeight, uint channels, int factor);


#endif //EDIX_SCALE_CUH
