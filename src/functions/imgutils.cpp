//
// Created by f3m on 19/01/24.
//


#include "imgutils.hpp"

unsigned char *from1To3Channels(unsigned char *imgIn, uint width, uint height)
{
    auto *imgOut = (unsigned char *) malloc(width * height * 3 * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore nell'allocazione della memoria di ImgOut");
        return nullptr;
    }

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int k = 0; k < 3; ++k)
                imgOut[(x + y * width) * 3 + k] = imgIn[x + y * width];

    free(imgIn);
    return imgOut;
}

unsigned char *loadImage(char *path, uint *width, uint *height, uint *channels)
{
    char *copy = strdup(path);
    char *extension = strtok(copy, ".");
    char *lastToken;

    while ((lastToken = strtok(nullptr, ".")) != nullptr)
        extension = lastToken;

    unsigned char *img;

    if (strcmp(extension, "ppm") == 0)
    {
        img = loadPPM(path, width, height, channels);
    } else if (strcmp(extension, "png") == 0)
    {
        img = loadPng(path, width, height, channels);
    } else if (strcmp(extension, "jpeg") == 0 || strcmp(extension, "jpg") == 0)
    {
        img = loadJpeg(path, width, height, channels);
    } else
    {
        free(copy);
        E_Print("Formato immagine non valido!\n");
        return nullptr;
    }


    free(copy);
    return img;
}
int writeImage(const char *path, unsigned char *img, uint width, uint height, uint channels)
{
    char *copy = strdup(path);
    char *extension = strtok(copy, ".");
    char *lastToken;

    while ((lastToken = strtok(nullptr, ".")) != nullptr)
        extension = lastToken;

    if (strcmp(extension, "ppm") == 0)
        writePPM(path, img, width, height, channels);
    else if (strcmp(extension, "jpeg") == 0 || strcmp(extension, "jpg") == 0)
        writeJpeg(path, img, width, height, 75, channels);
    else
    {
        free(copy);
        E_Print("Formato immagine non valido!\n");
        return 1;
    }


    free(copy);
    return 0;
}


unsigned char *loadPPM(const char *path, uint *width, uint *height, uint *channels)
{
    FILE *file = fopen(path, "rb");

    if (!file)
    {
        E_Print("Failed to open file %s\n", path);
        return nullptr;
    }

    char header[3];
    fscanf(file, "%2s", header);
    if (header[0] == 'P' && header[1] == '5')
    {
        *channels = 1;
    } else if (header[0] == 'P' && header[1] == '6')
    {
        *channels = 3;
    } else
    {
        fclose(file);
        E_Print("Invalid PPM file\n");
        return nullptr;
    }


    char sWidth[10];
    char sHeight[10];
    fscanf(file, "%s %s", sWidth, sHeight);

    *width = strtol(sWidth, nullptr, 10);
    *height = strtol(sHeight, nullptr, 10);
    if (*width <= 0 || *height <= 0)
    {
        fclose(file);
        E_Print("Invalid PPM file\n");
        return nullptr;
    }


    char ignored[4];
    fscanf(file, "%s", ignored);
    fgetc(file);  // Skip single whitespace character

    size_t iSize = *width * *height * *channels;

    auto *img = (unsigned char *) malloc(iSize * sizeof(unsigned char));
    if (!img)
    {
        fclose(file);
        E_Print("Failed to allocate memory\n");
        return nullptr;
    }

    fread(img, sizeof(unsigned char), iSize, file);
    fclose(file);

    return img;
}
unsigned char *loadJpeg(const char *path, uint *width, uint *height, uint *channels)
{
    FILE *jpeg_file = fopen(path, "rb");
    if (!jpeg_file)
    {
        E_Print("Errore nell'apertura del file JPEG.\n");
        return nullptr;
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
    *width = (uint) cinfo.output_width;
    *height = (uint) cinfo.output_height;
    int num_components = cinfo.output_components;
    *channels = num_components;

    // Allocazione di buffer per i dati dell'immagine
    auto buffer = (JSAMPARRAY) malloc(sizeof(JSAMPROW) * *height);
    for (int i = 0; i < *height; ++i)
        buffer[i] = (JSAMPROW) malloc(sizeof(JSAMPLE) * *width * num_components);

    // Lettura dei dati dell'immagine
    while (cinfo.output_scanline < cinfo.output_height)
        jpeg_read_scanlines(&cinfo, buffer + cinfo.output_scanline, cinfo.output_height - cinfo.output_scanline);

    // Chiusura della struttura JPEG
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(jpeg_file);

    // Allocazione di un array per i dati dell'immagine PPM
    auto *img = (unsigned char *) malloc(*width * *height * num_components * sizeof(unsigned char));

    // Copia i dati dell'immagine nel nuovo array PPM
    for (int i = 0; i < *height; ++i)
        for (int j = 0; j < *width; ++j)
            for (int k = 0; k < num_components; ++k)
                img[i * *width * 3 + j * 3 + k] = buffer[i][j * num_components + k];

    // Libera la memoria allocata per il buffer temporaneo
    for (int i = 0; i < *height; ++i)
        free(buffer[i]);
    free(buffer);


    return img;
}
unsigned char *loadPng(const char *path, uint *width, uint *height, uint *channels)
{
    FILE *png_file = fopen(path, "rb");
    if (!png_file)
    {
        E_Print("Errore nell'apertura del file PNG.\n");
        return nullptr;
    }

    // Inizializzazione della struttura PNG
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr)
    {
        fclose(png_file);
        E_Print("Errore nell'inizializzazione della struttura PNG.\n");
        return nullptr;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct(&png_ptr, (png_infopp) nullptr, (png_infopp) nullptr);
        fclose(png_file);
        E_Print("Errore nell'inizializzazione della struttura di informazioni PNG.\n");
        return nullptr;
    }

    // Impostazione della gestione degli errori durante la lettura del PNG
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        E_Print("Errore durante la lettura del file PNG.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) nullptr);
        fclose(png_file);
        return nullptr;
    }

    // Inizializzazione della lettura del PNG da file
    png_init_io(png_ptr, png_file);

    // Leggi l'intestazione del PNG
    png_read_info(png_ptr, info_ptr);

    // Recupera le informazioni sull'immagine
    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    *channels = png_get_channels(png_ptr, info_ptr);
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
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) nullptr);
    fclose(png_file);

    // Copia i dati dell'immagine in un array di unsigned char
    auto *image_data = (unsigned char *) malloc(*width * *height * *channels);
    for (int y = 0; y < *height; y++)
        for (int x = 0; x < *width; x++)
            for (int k = 0; k < *channels; ++k)
                image_data[(y * *width + x) * *channels + k] = row_pointers[y][x * *channels + k];

    // Libera la memoria allocata per le righe
    for (int y = 0; y < *height; y++)
        free(row_pointers[y]);
    free(row_pointers);

    return image_data;
}

void writePPM(const char *path, unsigned char *img, uint width, uint height, uint channels)
{
    if (channels != 3 && channels != 1)
    {
        E_Print("Canali non validi!\n");
        return;
    }

    FILE *file = fopen(path, "w");
    if (!file)
    {
        E_Print("Failed to open file %s\n", path);
        return;
    }

    fprintf(file, "%s\n%d %d\n255\n", channels == 1 ? "P5" : "P6", width, height);

    fwrite(img, channels, width * height, file);

    fclose(file);
}
void writeJpeg(const char *path, unsigned char *img, uint width, uint height, int quality, uint channels)
{
    if (channels != 3 && channels != 1)
    {
        E_Print("Canali non validi!\n");
        return;
    }

    struct jpeg_compress_struct cinfo{};
    struct jpeg_error_mgr jerr{};
    FILE *outfile = fopen(path, "wb");
    if (!outfile)
    {
        E_Print("Errore nell'apertura del file di output.");
        return;
    }
    // Inizializzazione della struttura jpeg_compress_struct
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    // Impostazione dei parametri dell'immagine
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = (int) channels;  // Scala di grigi o RGB
    cinfo.in_color_space = JCS_RGB;

    // Impostazione del file di output
    jpeg_stdio_dest(&cinfo, outfile); // Passa direttamente il puntatore al file di output

    // Impostazione dei parametri di compressione
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    // Avvio della compressione
    jpeg_start_compress(&cinfo, TRUE);

    // Allocazione della memoria per una riga di dati RGB
    JSAMPROW row_pointer[1];
    int row_stride = (int) channels;

    // Scrittura delle righe dell'immagine
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = &img[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // Finalizzazione della compressione
    jpeg_finish_compress(&cinfo);

    // Pulizia delle risorse
    jpeg_destroy_compress(&cinfo);
}
