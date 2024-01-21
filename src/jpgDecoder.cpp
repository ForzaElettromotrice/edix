#include "utils.hpp"

unsigned char *jpeg_to_ppm(const char *jpeg_filename, int *width, int *height) {
    FILE *jpeg_file = fopen(jpeg_filename, "rb");
    if (!jpeg_file) {
        fprintf(stderr, "Errore nell'apertura del file JPEG.\n");
        exit(EXIT_FAILURE);
    }

    // Inizializzazione della struttura JPEG
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

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
    JSAMPARRAY buffer = (JSAMPARRAY)malloc(sizeof(JSAMPROW) * *height);
    for (int i = 0; i < *height; ++i) {
        buffer[i] = (JSAMPROW)malloc(sizeof(JSAMPLE) * *width * num_components);
    }

    // Lettura dei dati dell'immagine
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer + cinfo.output_scanline, cinfo.output_height - cinfo.output_scanline);
    }

    // Chiusura della struttura JPEG
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(jpeg_file);

    // Allocazione di un array per i dati dell'immagine PPM
    unsigned char *img = (unsigned char *)malloc(*width * *height * 3);

    // Copia i dati dell'immagine nel nuovo array PPM
    for (int i = 0; i < *height; ++i) {
        for (int j = 0; j < *width; ++j) {
            for (int k = 0; k < num_components; ++k) {
                img[i * *width * 3 + j * 3 + k] = buffer[i][j * num_components + k];
            }
        }
    }

    // Libera la memoria allocata per il buffer temporaneo
    for (int i = 0; i < *height; ++i) {
        free(buffer[i]);
    }
    free(buffer);

    // Creazione della struttura PPMImage da restituire
    //PPMImage ppm_image = {width, height, ppm_data};
  
    
    return img;
}