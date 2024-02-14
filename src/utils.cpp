//
// Created by f3m on 08/02/24.
//
#include "utils.hpp"


void E_Print(const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    fprintf(stderr, RED "Error: " RESET);
    vfprintf(stderr, msg, args);
    va_end(args);
}

void D_Print(const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    if (DEBUG)
    {
        printf(YELLOW "DEBUG: " RESET);
        vprintf(msg, args);
    }
    va_end(args);
}