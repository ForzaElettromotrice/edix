//
// Created by f3m on 11/01/24.
//

#ifndef EDIX_TEMP_H
#define EDIX_TEMP_H

#include <stdio.h>
#include <stdlib.h>
#include <libpq-fe.h>
#include <hiredis/hiredis.h>

enum Modex_t{
    IMMEDIATE,
    PROGRAMMED
};
enum Tup_t{
    BILINEAR,
    BICUBIC
};
enum Compx_t{
    JPEG,
    PNG,
    PPM
};
enum Tppx_t{
    SERIAL,
    OMP,
    CUDA
};

int get_from_settings(char *projectName);


#endif //EDIX_TEMP_H
