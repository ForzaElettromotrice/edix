//
// Created by f3m on 30/01/24.
//

#ifndef EDIX_OVERLAP_CUH
#define EDIX_OVERLAP_CUH

#include <iostream>
#include "../utils.hpp"
#include "sys/mman.h"

__global__ void overlap(const unsigned char *imgIn, unsigned char *imgOut, uint iWidth, uint iHeight, uint channels1, uint oWidth, uint oHeight, uint x, uint y);

#endif //EDIX_OVERLAP_CUH
