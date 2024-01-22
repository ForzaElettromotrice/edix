//
// Created by f3m on 19/01/24.
//

#ifndef EDIX_UPSCALE_CUH
#define EDIX_UPSCALE_CUH

#include "functions.cuh"

int bilinearInterpolation(int p00, int p01, int p10, int p11, double alpha, double beta);
double cubicInterpolate(double A, double B, double C, double D, double t);

#endif //EDIX_UPSCALE_CUH
