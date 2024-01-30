//
// Created by f3m on 30/01/24.
//

#ifndef EDIX_OVERLAP_CUH
#define EDIX_OVERLAP_CUH

#include "functions.cuh"
#include "sys/mman.h"

__global__ void overlap(unsigned char *img, const unsigned char *img2, uint width, uint width2, uint height2, uint posX, uint posY);


#endif //EDIX_OVERLAP_CUH
