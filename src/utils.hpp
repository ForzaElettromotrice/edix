//
// Created by f3m on 30/12/23.
//

#ifndef EDIX_UTILS_HPP
#define EDIX_UTILS_HPP

#define DEBUG 1
#define D_PRINT(format, ...) \
        if(DEBUG)            \
            printf(YELLOW "DEBUG: " RESET format, ##__VA_ARGS__)
#define handle_error(msg) \
    fprintf(stderr, msg);\
    return(EXIT_FAILURE)

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define YELLOW  "\033[1;33m"
#define BLUE    "\033[1;34m"
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
