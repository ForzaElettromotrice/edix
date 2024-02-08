//
// Created by f3m on 19/01/24.
//

#ifndef EDIX_COMPOSITION_CUH
#define EDIX_COMPOSITION_CUH

#include <iostream>
#include "../utils.hpp"

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

int copyMatrix(const unsigned char *mIn, unsigned char *mOut, uint widthI, uint heightI, uint widthO, uint channels1, uint channels2, uint x, uint y);
int copyMatrixOmp(const unsigned char *mIn, unsigned char *mOut, uint widthI, uint heightI, uint widthO, uint x, uint y, int nThread);


#endif //EDIX_COMPOSITION_CUH
