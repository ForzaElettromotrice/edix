#ifndef EDIX_GRAYSCALE_CUH
#define EDIX_GRAYSCALE_CUH

#include "functions.cuh"
#include <sys/mman.h>


__global__ void grayscaleCUDA(unsigned char *ImgIn, unsigned char *ImgOut, uint width, uint height);

#endif //EDIX_GRAYSCALE_CUH