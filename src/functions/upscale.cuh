//
// Created by f3m on 19/01/24.
//

#ifndef EDIX_UPSCALE_CUH
#define EDIX_UPSCALE_CUH

#include "functions.cuh"
#include <sys/mman.h>


int bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta);
double cubicInterpolate(double A, double B, double C, double D, double t);
void createSquare(unsigned char square[16][3], const unsigned char *img, int x, int y, uint width, uint height, uint channels);
__global__ void bilinearUpscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, int factor);
__global__ void bicubicUpscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, int factor, uint channels);
__device__ void createSquareDEVICE(unsigned char square[16][3], const unsigned char *img, int x, int y, uint width, uint height, uint channels);
__device__ double cubicInterpolateDEVICE(double A, double B, double C, double D, double t);


#endif //EDIX_UPSCALE_CUH
