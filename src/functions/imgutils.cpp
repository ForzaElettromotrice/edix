//
// Created by f3m on 19/01/24.
//

#include "functions.hu"

unsigned char *loadPPM(const char *path, int *width, int *height)
{
    FILE *file = fopen(path, "rb");

    if (!file)
    {
        fprintf(stderr, "Failed to open file\n");
        return nullptr;
    }

    char header[3];
    fscanf(file, "%2s", header);
    if (header[0] != 'P' || header[1] != '6')
    {
        fprintf(stderr, "Invalid PPM file\n");
        return nullptr;
    }

    fscanf(file, "%d %d", width, height);

    int maxColor;
    fscanf(file, "%d", &maxColor);

    fgetc(file);  // Skip single whitespace character

    unsigned char *img = (unsigned char *) malloc((*width) * (*height) * CHANNELS);
    if (!img)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        return NULL;
    }

    fread(img, CHANNELS, *width * *height, file);

    fclose(file);

    return img;
}

void writePPM(const char *path, unsigned char *img, int width, int height, const char *format)
{
    FILE *file = fopen(path, "wb");

    if (!file)
    {
        fprintf(stderr, "Failed to open file\n");
        return;
    }

    fprintf(file, "%s\n%d %d\n255\n", format, width, height);

    fwrite(img, 3, width * height, file);

    fclose(file);
}