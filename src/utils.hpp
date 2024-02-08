//
// Created by f3m on 30/12/23.
//

#ifndef EDIX_UTILS_HPP
#define EDIX_UTILS_HPP

#include <iostream>
#include <cstdlib>
#include <cstdarg>
#include <jpeglib.h>
#include <png.h>

#define DEBUG 1

#define RED       "\033[31m"
#define GREEN     "\033[32m"
#define YELLOW    "\033[33m"
#define BLUE      "\033[34m"
#define BOLD      "\033[1m"
#define ITALIC    "\033[3m"
#define UNDERLINE "\033[4m"
#define RESET     "\033[0m"

typedef enum
{
    HOMEPAGE,
    PROJECT,
    SETTINGS,
    EXIT
} Env;


void E_Print(const char *msg, ...);
void D_Print(const char *msg, ...);


#endif //EDIX_UTILS_HPP
