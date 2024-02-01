#ifndef EDIX_TEST_HPP
#define EDIX_TEST_HPP

#include <iostream>
#include <cstdint>
#include "../src/functions/functions.cuh"
#include <chrono>

double percentage(long time1, long time2);
long print_times(auto time1, auto time2);
void print_perc(double perc);
void test(unsigned char *img1, unsigned char *img2, const uint *width1, const uint *height1, const uint *width2, const uint *height2);

#endif

