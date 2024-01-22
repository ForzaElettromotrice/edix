#include "functions.cuh"

int parseGrayscaleArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");

    if (imgIn == nullptr || pathOut == nullptr)
    {
        handle_error("Errore nel parsing degli argomenti\n");
    }
    // TODO: leggere le immagini in base all'estenzione

    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;

    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        grayscaleSerial(img, pathOut, width, height);
        free(img);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        grayscaleOmp(img, pathOut, width, height);
        free(img);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        grayscaleCuda(img, pathOut, width, height);
        free(img);
    } else
    {
        free(tpp);
        handle_error("Errore nel parsing degli argomenti\n");
    }
    free(tpp);
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

    auto start = std::chrono::high_resolution_clock::now();

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

    auto end = std::chrono::high_resolution_clock::now();

    auto deltaOmp = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Serial %ld us\n", deltaOmp.count());

    writePPM(pathOut, img_out, width, height, "P5");
    return 0;
}
int grayscaleOmp(const unsigned char *imgIn, char *pathOut, uint width, uint height)
{
    auto *img_out = (unsigned char *) malloc((width * height) * sizeof(unsigned char));

    if (img_out == nullptr)
    {
        handle_error("Errore nell'allocazione della memoria");
    }

    int r, g, b, grayValue;

    auto start = std::chrono::high_resolution_clock::now();
//TODO: studiare il numero di thread ottimale per eseguire questo for
//TODO: mettere la schedule
#pragma omp parallel for collapse(2) num_threads(4) default(none) private(r, g, b, grayValue) shared(img_out, imgIn, width, height)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            r = imgIn[((y * width) + x) * 3];
            g = imgIn[((y * width) + x) * 3 + 1];
            b = imgIn[((y * width) + x) * 3 + 2];
            grayValue = (r + g + b) / CHANNELS;

            img_out[y * width + x] = grayValue;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto deltaOmp = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("omp %ld us\n", deltaOmp.count());

    writePPM(pathOut, img_out, width, height, "P5");

    return 0;
}
int grayscaleCuda(unsigned char *imgIn, char *pathOut, uint width, uint height)
{
    return 0;
}

