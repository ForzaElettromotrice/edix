//
// Created by f3m on 30/12/23.
//

#ifndef EDIX_UTILS_HPP
#define EDIX_UTILS_HPP

#include <cstdlib>

#define DEBUG 1
#define D_PRINT(format, ...) \
        if(DEBUG)            \
            printf(YELLOW "DEBUG: " RESET format, ##__VA_ARGS__)
#define handle_error(msg, ...) \
    fprintf(stderr, msg, ##__VA_ARGS__);\
    return(EXIT_FAILURE)

#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define BOLD    "\033[1m"
#define ITALIC "\033[3m"
#define RESET   "\033[0m"

typedef enum
{
    HOMEPAGE,
    PROJECT,
    SETTINGS,
    EXIT
} Env;

#endif //EDIX_UTILS_HPP
