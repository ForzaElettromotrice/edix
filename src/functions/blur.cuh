//
// Created by f3m on 30/01/24.
//

#ifndef EDIX_BLUR_CUH
#define EDIX_BLUR_CUH

#include "functions.cuh"
#include <sys/mman.h>


__global__ void blurShared(const unsigned char *img, unsigned char *blur_img, uint width, uint height, int radius);


#endif //EDIX_BLUR_CUH
