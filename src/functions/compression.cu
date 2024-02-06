//
// Created by f3m on 20/01/24.
//

#include "compression.cuh"

void saveIntIntoChar(unsigned char *img, size_t pos, int value)
{
    char a = (char) (value & 0xff);
    char b = (char) ((value >> 8) & 0xff);
    char c = (char) ((value >> 16) & 0xff);
    char d = (char) ((value >> 24) & 0xff);

    img[pos++] = d;
    img[pos++] = c;
    img[pos++] = b;
    img[pos] = a;
}
uint getIntFromChar(const unsigned char *img, size_t pos)
{
    uint out = 0;

    for (int i = 0; i < 4; ++i)
    {
        out <<= 8;
        out += (uint) img[pos++];
    }

    return out;
}


triple_t *lzTriples(unsigned char *img, size_t maxSize, size_t *oSize, int n)
{
    size_t lSize = 258;
    size_t bSize = 32 * 1024 - 258;
    unsigned char *look = img;


    size_t tSize = 1000;
    size_t tLen = 0;
    auto *triples = (triple_t *) malloc(tSize * sizeof(triple_t));

    triple_t triple;
    triple.o = 0;
    triple.l = 0;
    triple.a = *look;
    triples[tLen++] = triple;
    look++;

    while (look < img + maxSize)
    {
        triple.o = 0;
        triple.l = 0;
        triple.a = *look;

        for (int i = 1; i < bSize; ++i)
        {
            if (look - i < img)
                break;

            for (int j = 0; j < lSize; ++j)
            {

                if (*(look - i + j) == *(look + j))
                {
                    if (j == lSize - 1)
                    {
                        triple.o = i;
                        triple.l = j;
                        triple.a = '\0';
                        break;
                    }
                    continue;
                }
                if (j == 0)
                    break;
                if (j > triple.l)
                {
                    if (j <= n)
                        break;
                    triple.o = i;
                    triple.l = j;
                    triple.a = '\0';
                    break;
                }

                break;
            }
        }
        if (tLen >= tSize - 1)
        {
            tSize *= 2;
            auto *tmp = (triple_t *) realloc(triples, tSize * sizeof(triple_t));
            if (tmp == nullptr)
            {
                free(triples);
                fprintf(stderr, RED "Error: " RESET "Error while realloc!\n");
                return nullptr;
            }
            triples = tmp;
        }


        triples[tLen++] = triple;
        look += triple.l == 0 ? 1 : triple.l;
    }


    *oSize = tLen;


    return triples;
}
unsigned char *lzTransformer(unsigned char *img, size_t maxSize, size_t *oSize, int n)
{
    triple_t *triples = lzTriples(img, maxSize, oSize, n);
    size_t iSize = *oSize * 9;
    size_t iLen = 0;
    auto *oImg = (unsigned char *) malloc(iSize * sizeof(unsigned char *));

    /**
     *  Caratteri usati come flag:              <p>
     *      0xca    ->  caratteri               <p>
     *      0xac    ->  numeri                  <p>
     *
     *  Verrà codificato come segue:            <p>
     *  - Prima di una seguenza di caratteri    <p>
     *  ----- 1 byte -> flag caratteri          <p>
     *  ----- 4 byte -> unsiged intero che indica la lunghezza della sequenza di caratteri  <p>
     *  ----- n byte -> caratteri               <p>
     *  - Prima di una sequenza di numeri       <p>
     *  ----- 1 byte -> flag numeri             <p>
     *  ----- 4 byte -> offset                  <p>
     *  ----- 4 byte -> lunghezza               <p>
     */

    size_t tSize = iSize;
    auto *tmp = (unsigned char *) malloc(tSize * sizeof(unsigned char *));

    int i = 0;
    int j;
    while (i < *oSize)
    {
        triple_t triple = triples[i];
        if (triple.o == 0)
        {
            j = 0;
            while (triple.o == 0)
            {
                if (j >= tSize)
                {
                    tSize *= 2;
                    auto *tmp1 = (unsigned char *) realloc(tmp, tSize * sizeof(unsigned char));
                    if (tmp1 == nullptr)
                    {
                        free(oImg);
                        free(tmp);
                        fprintf(stderr, RED "Error: " RESET "Error while realloc!\n");
                        return nullptr;
                    }
                    tmp = tmp1;
                }
                tmp[j++] = triple.a;
                triple = triples[++i];
                if (i == *oSize)
                    break;
            }
            i--;
            if (iLen + 4 + j >= iSize)
            {
                iSize *= 2;
                auto *tmp1 = (unsigned char *) realloc(oImg, iSize * sizeof(unsigned char));
                if (tmp == nullptr)
                {
                    free(oImg);
                    free(tmp);
                    fprintf(stderr, RED "Error: " RESET "Error while realloc!\n");
                    return nullptr;
                }
                oImg = tmp1;
            }
            oImg[iLen++] = 'c';
            saveIntIntoChar(oImg, iLen, j);
            iLen += 4;
            memcpy(oImg + iLen, tmp, j);
            iLen += j;
        } else
        {
            if (iLen + 9 >= iSize)
            {
                iSize *= 2;
                auto *tmp1 = (unsigned char *) realloc(oImg, iSize * sizeof(unsigned char));
                if (tmp == nullptr)
                {
                    free(oImg);
                    free(tmp);
                    fprintf(stderr, RED "Error: " RESET "Error while realloc!\n");
                    return nullptr;
                }
                oImg = tmp1;
            }
            oImg[iLen++] = 'i';
            saveIntIntoChar(oImg, iLen, triple.o);
            saveIntIntoChar(oImg, iLen + 4, triple.l);
            iLen += 8;
        }
        i++;
    }

    free(tmp);
    free(triples);

    *oSize = iLen;

    return oImg;
}

unsigned char *decoder(const unsigned char *img, size_t maxSize, size_t *oSize)
{
    size_t iSize = maxSize * 2;
    size_t iLen = 0;
    auto *oImg = (unsigned char *) malloc(iSize * sizeof(unsigned char));

    uint i = 0;
    while (i < maxSize)
    {
        if (img[i] == 'c')
        {
            uint len = getIntFromChar(img, ++i);
            i += 4;

            if (iLen + len >= iSize)
            {
                iSize *= 2;
                auto *tmp = (unsigned char *) realloc(oImg, iSize * sizeof(unsigned char));
                if (tmp == nullptr)
                {
                    free(oImg);
                    fprintf(stderr, RED "Error: " RESET "Error while realloc!\n");
                    return nullptr;
                }
                oImg = tmp;
            }

            for (int j = 0; j < len; ++j)
                oImg[iLen++] = img[i++];

        } else if (img[i] == 'i')
        {
            uint o = getIntFromChar(img, ++i);
            uint l = getIntFromChar(img, i + 4);
            i += 8;     //8 perché ho letto 2 interi, 1 per il prossimo ciclo del while

            if (iLen + l >= iSize)
            {
                iSize *= 2;
                auto *tmp = (unsigned char *) realloc(oImg, iSize * sizeof(unsigned char));
                if (tmp == nullptr)
                {
                    free(oImg);
                    fprintf(stderr, RED "Error: " RESET "Error while realloc!\n");
                    return nullptr;
                }
                oImg = tmp;
            }

            uint start = iLen;
            for (int j = 0; j < l; ++j)
                oImg[iLen++] = oImg[start - o + j];
        } else
        {
            *oSize = 0;
            free(oImg);
            fprintf(stderr, RED "Error: " RESET "Errore, immagine compressa corrotta!\n");
            return nullptr;
        }
    }

    *oSize = iLen;
    return oImg;
}

//TODO: separare gli rgb e fare dei chunk