//
// Created by f3m on 19/01/24.
//

#ifndef EDIX_DOWNSCALE_CUH
#define EDIX_DOWNSCALE_CUH

#include "functions.cuh"
#include "upscale.cuh"

__global__ void bilinearDownscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut,uint width, uint height, int factor);

#endif //EDIX_DOWNSCALE_CUH
