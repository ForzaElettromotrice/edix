#include "functions.hu"

int parseBlurArgs(char *args)
{
    return 0;
}

int grayScaleSerial(char *pathIn, char *pathOut)
{
    int width, height;
    int gray_value, r, g, b, i = 0;
    unsigned char *img_in = loadPPM(pathIn, &width, &height),
            *img_out = (unsigned char *) malloc((width * height) * sizeof(unsigned char));

    if (img_out == nullptr)
    {
        handle_error("Errore nell'allocazione della memoria\n");
    }

    for (int y = 0; y < height; y += 1)
    {
        for (int x = 0; x < width; x += 1)
        {
            // prendi i valori di tre pixel contigui
            r = img_in[((y * width) + x) * 3];
            g = img_in[((y * width) + x) * 3 + 1];
            b = img_in[((y * width) + x) * 3 + 2];
            // Fai la media per prendere il grigio
            gray_value = (r + g + b) / CHANNELS;
            // Inseriscilo come primo pixel di img_out
            img_out[i++] = gray_value;
        }
    }

    writePPM(pathOut, img_out, width, height, "P5");
    free(img_in);
    free(img_out);
    return 0;
}
int grayScaleOmp(char *pathIn, char *pathOut)
{
    return 0;
}
int grayScaleCuda(char *pathIn, char *pathOut)
{
    return 0;
}

