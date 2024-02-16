//
// Created by f3m on 31/01/24.
//

#ifndef EDIX_IMGUTILS_HPP
#define EDIX_IMGUTILS_HPP

#include <iostream>
#include <cstring>
#include <jpeglib.h>
#include <png.h>
#include "../utils.hpp"

unsigned char *from1To3Channels(unsigned char *imgIn, uint width, uint height);

unsigned char *loadImage(char *path, uint *width, uint *height, uint *channels);
int writeImage(const char *path, unsigned char *img, uint width, uint height, uint channels);


unsigned char *loadPPM(const char *path, uint *width, uint *height, uint *channels);
unsigned char *loadJpeg(const char *path, uint *width, uint *height, uint *channels);
unsigned char *loadPng(const char *path, uint *width, uint *height, uint *channels);

void writePPM(const char *path, unsigned char *img, uint width, uint height, uint channels);
void writeJpeg(const char *path, unsigned char *img, uint width, uint height, int quality, uint channels);
#endif //EDIX_IMGUTILS_HPP
