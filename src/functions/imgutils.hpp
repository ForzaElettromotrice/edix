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

unsigned char *loadImage(char *path, uint *width, uint *height, char format[3]);


unsigned char *loadPPM(const char *path, uint *width, uint *height, char format[3]);
unsigned char *loadJpeg(const char *path, uint *width, uint *height, char format[3]);
unsigned char *loadPng(const char *path, uint *width, uint *height, char format[3]);

void writePPM(const char *path, unsigned char *img, uint width, uint height, const char *format);
void writeJpeg(const char *path, unsigned char *img, uint width, uint height, int quality, const char format[3]);
#endif //EDIX_IMGUTILS_HPP
