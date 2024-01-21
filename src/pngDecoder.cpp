#include "utils.hpp"

unsigned char *decode_png(const char *filename, int *width, int *height) {
    FILE *png_file = fopen(filename, "rb");
    if (!png_file) {
        fprintf(stderr, "Errore nell'apertura del file PNG.\n");
        exit(EXIT_FAILURE);
    }

    // Inizializzazione della struttura PNG
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Errore nell'inizializzazione della struttura PNG.\n");
        fclose(png_file);
        exit(EXIT_FAILURE);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Errore nell'inizializzazione della struttura di informazioni PNG.\n");
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        fclose(png_file);
        exit(EXIT_FAILURE);
    }

    // Impostazione della gestione degli errori durante la lettura del PNG
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Errore durante la lettura del file PNG.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
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
    png_bytep *row_pointers = (png_bytep*)malloc(*height * sizeof(png_bytep));
    for (int y = 0; y < *height; y++)
        row_pointers[y] = (png_byte*)malloc(row_bytes);

    // Lettura dell'immagine
    png_read_image(png_ptr, row_pointers);

    // Chiusura della struttura PNG
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    fclose(png_file);

    // Copia i dati dell'immagine in un array di unsigned char
    unsigned char *image_data = (unsigned char *)malloc(*width * *height * 4);
    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {
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