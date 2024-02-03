//
// Created by f3m on 19/01/24.
//

#ifndef EDIX_DOWNSCALE_CUH
#define EDIX_DOWNSCALE_CUH

#include "functions.cuh"
#include "upscale.cuh"

void createSquareD(unsigned char square[16][3], const unsigned char *img, int x, int y, uint width, uint height, uint channels);
__global__ void bilinearDownscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, int factor);
__global__ void bicubicDownscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, int factor, uint channels);

#endif //EDIX_DOWNSCALE_CUH
