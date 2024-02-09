#ifndef EDIX_GRAYSCALE_CUH
#define EDIX_GRAYSCALE_CUH

#include <iostream>
#include "../utils.hpp"
#include <sys/mman.h>


__global__ void grayscale(const unsigned char *ImgIn, unsigned char *ImgOut, uint width, uint height);

#endif //EDIX_GRAYSCALE_CUH