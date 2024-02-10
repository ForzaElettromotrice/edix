#ifndef EDIX_TEST_HPP
#define EDIX_TEST_HPP

#include <iostream>
#include <chrono>
#include "../src/functions/parser.hpp"
#include "../src/utils.hpp"


double percentage(long time1, long time2);
long print_times(auto time1, auto time2);
void print_perc(double perc);
void testAccuracy(const unsigned char *bigImage, const unsigned char *smallImage, const unsigned char *grayImage, uint width1, uint height1, uint width2, uint height2, uint width3, uint height3);
#endif //EDIX_TEST_HPP

