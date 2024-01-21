//
// Created by f3m on 20/01/24.
//

#ifndef EDIX_COMPRESSION_CUH
#define EDIX_COMPRESSION_CUH

#include "../utils.hpp"

typedef struct
{
    int o;
    int l;
    unsigned char a;
} triple_t;


//UTILS
void saveIntIntoChar(unsigned char *img, size_t pos, int value);
uint getIntFromChar(const unsigned char *img, size_t pos);


triple_t *lzTriples(unsigned char *img, size_t maxSize, size_t *oSize, int n);
unsigned char *lzTransformer(unsigned char *img, size_t maxSize, size_t *oSize, int n);

unsigned char *decoder(const unsigned char *img, size_t maxSize, size_t *oSize);


unsigned char *chunkEncoder(unsigned char *img, uint widht, uint height, size_t cSize);


#endif //EDIX_COMPRESSION_CUH
