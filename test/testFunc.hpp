#ifndef EDIX_TEST_HPP
#define EDIX_TEST_HPP

#include <iostream>
#include <cstdint>
#include <chrono>

double percentage(long time1, long time2);
long print_times(auto time1, auto time2);
void print_perc(double perc);
void test(const unsigned char *img1, const unsigned char *img2, const uint *width1, const uint *height1, const uint *width2, const uint *height2, const uint *channels1, const uint *channels2);

#endif

