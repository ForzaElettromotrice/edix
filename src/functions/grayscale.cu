#include "functions.hu"

int parseBlurArgs(char *args)
{
    return 0;
}

int grayscaleSerial(const unsigned char *imgIn, char *pathOut, uint width, uint height)
{
    auto *img_out = (unsigned char *) malloc((width * height) * sizeof(unsigned char));

    if (img_out == nullptr)
    {
        handle_error("Errore nell'allocazione della memoria\n");
    }

    int r;
    int g;
    int b;
    int i = 0;
    int grayValue;

    for (int y = 0; y < height; y += 1)
    {
        for (int x = 0; x < width; x += 1)
        {
            // prendi i valori di tre pixel contigui
            r = imgIn[((y * width) + x) * 3];
            g = imgIn[((y * width) + x) * 3 + 1];
            b = imgIn[((y * width) + x) * 3 + 2];
            // Fai la media per prendere il grigio
            grayValue = (r + g + b) / CHANNELS;
            // Inseriscilo come primo pixel di img_out
            img_out[i++] = grayValue;
        }
    }
    writePPM(pathOut, img_out, width, height, "P5");
    return 0;
}
int grayscaleOmp(unsigned char *imgIn, char *pathOut, uint width, uint height)
{
    return 0;
}
int grayscaleCuda(unsigned char *imgIn, char *pathOut, uint width, uint height)
{
    return 0;
}

