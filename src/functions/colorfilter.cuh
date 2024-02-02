#ifndef EDIX_COLORFILTER_CUH
#define EDIX_COLORFILTER_CUH

#include "functions.cuh"
#include <sys/mman.h>


__global__ void colorFilterCUDA(unsigned char *imgIn,unsigned char *imgOut, uint width, uint height, uint r, uint g, uint b, uint tolerance);


#endif //EDIX_COLORFILTER_CUH