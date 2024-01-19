
#ifndef EDIX_GRAYSCALE_HU
#define EDIX_GRAYSCALE_HU

#include <iostream>
#include "utils.hpp"
#include <cstdlib>

#define CHANNELS 3

//int grayScaleCuda();
//int grayScaleOmp();
int grayScaleSerial(char *pathIn, char *pathOut);

// utils
unsigned char* loadPPM(const char* path, int* width, int* height);
//TODO : PASSA COME PARAMETRO IL TIPO DI PPM (P6,P5)
void writePPM(const char* path, unsigned char* img, int width, int height);

#endif  //EDIX_GRAYSCALE_HU
