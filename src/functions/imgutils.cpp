//
// Created by f3m on 19/01/24.
//

#include "functions.cuh"

unsigned char *loadPPM(const char *path, uint *width, uint *height)
{
    FILE *file = fopen(path, "rb");

    if (!file)
    {
        fprintf(stderr, "Failed to open file %s\n", path);
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

    auto *img = (unsigned char *) malloc((*width) * (*height) * CHANNELS);
    if (!img)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        return nullptr;
    }

    fread(img, CHANNELS, *width * *height, file);

    fclose(file);

    return img;
}

void writePPM(const char *path, unsigned char *img, uint width, uint height, const char *format)
{
    FILE *file = fopen(path, "wb");

    if (!file)
    {
        fprintf(stderr, "Failed to open file %s\n", path);
        return;
    }

    fprintf(file, "%s\n%d %d\n255\n", format, width, height);

    fwrite(img, 3, width * height, file);

    fclose(file);
}

unsigned char *jpegDecode(const char *path, int *width, int *height)
{
    FILE *jpeg_file = fopen(path, "rb");
    if (!jpeg_file)
    {
        fprintf(stderr, "Errore nell'apertura del file JPEG.\n");
        exit(EXIT_FAILURE);
    }

    // Inizializzazione della struttura JPEG
    struct jpeg_decompress_struct cinfo{};
    struct jpeg_error_mgr jerr{};

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, jpeg_file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    // Informazioni sull'immagine
    *width = (int) cinfo.output_width;
    *height = (int) cinfo.output_height;
    int num_components = cinfo.output_components;

    // Allocazione di buffer per i dati dell'immagine
    auto buffer = (JSAMPARRAY) malloc(sizeof(JSAMPROW) * *height);
    for (int i = 0; i < *height; ++i)
    {
        buffer[i] = (JSAMPROW) malloc(sizeof(JSAMPLE) * *width * num_components);
    }

    // Lettura dei dati dell'immagine
    while (cinfo.output_scanline < cinfo.output_height)
    {
        jpeg_read_scanlines(&cinfo, buffer + cinfo.output_scanline, cinfo.output_height - cinfo.output_scanline);
    }

    // Chiusura della struttura JPEG
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(jpeg_file);

    // Allocazione di un array per i dati dell'immagine PPM
    auto *img = (unsigned char *) malloc(*width * *height * 3);

    // Copia i dati dell'immagine nel nuovo array PPM
    for (int i = 0; i < *height; ++i)
    {
        for (int j = 0; j < *width; ++j)
        {
            for (int k = 0; k < num_components; ++k)
            {
                img[i * *width * 3 + j * 3 + k] = buffer[i][j * num_components + k];
            }
        }
    }

    // Libera la memoria allocata per il buffer temporaneo
    for (int i = 0; i < *height; ++i)
    {
        free(buffer[i]);
    }
    free(buffer);


    return img;
}

unsigned char *pngDecode(const char *path, int *width, int *height)
{
    FILE *png_file = fopen(path, "rb");
    if (!png_file)
    {
        fprintf(stderr, "Errore nell'apertura del file PNG.\n");
        exit(EXIT_FAILURE);
    }

    // Inizializzazione della struttura PNG
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr)
    {
        fprintf(stderr, "Errore nell'inizializzazione della struttura PNG.\n");
        fclose(png_file);
        exit(EXIT_FAILURE);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        fprintf(stderr, "Errore nell'inizializzazione della struttura di informazioni PNG.\n");
        png_destroy_read_struct(&png_ptr, (png_infopp) nullptr, (png_infopp) nullptr);
        fclose(png_file);
        exit(EXIT_FAILURE);
    }

    // Impostazione della gestione degli errori durante la lettura del PNG
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        fprintf(stderr, "Errore durante la lettura del file PNG.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) nullptr);
        fclose(png_file);
        exit(EXIT_FAILURE);
    }

    // Inizializzazione della lettura del PNG da file
    png_init_io(png_ptr, png_file);

    // Leggi l'intestazione del PNG
    png_read_info(png_ptr, info_ptr);

    // Recupera le informazioni sull'immagine
    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Imposta la lettura RGBA automatica
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);

    // Aggiorna le informazioni dopo le trasformazioni
    png_read_update_info(png_ptr, info_ptr);

    // Allocazione dell'array di output
    size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    auto *row_pointers = (png_bytep *) malloc(*height * sizeof(png_bytep));
    for (int y = 0; y < *height; y++)
        row_pointers[y] = (png_byte *) malloc(row_bytes);

    // Lettura dell'immagine
    png_read_image(png_ptr, row_pointers);

    // Chiusura della struttura PNG
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) NULL);
    fclose(png_file);

    // Copia i dati dell'immagine in un array di unsigned char
    auto *image_data = (unsigned char *) malloc(*width * *height * 4);
    for (int y = 0; y < *height; y++)
    {
        for (int x = 0; x < *width; x++)
        {
            image_data[(y * *width + x) * 4 + 0] = row_pointers[y][x * 4 + 0];
            image_data[(y * *width + x) * 4 + 1] = row_pointers[y][x * 4 + 1];
            image_data[(y * *width + x) * 4 + 2] = row_pointers[y][x * 4 + 2];
            image_data[(y * *width + x) * 4 + 3] = row_pointers[y][x * 4 + 3];
        }
    }

    // Libera la memoria allocata per le righe
    for (int y = 0; y < *height; y++)
        free(row_pointers[y]);
    free(row_pointers);

    return image_data;
}