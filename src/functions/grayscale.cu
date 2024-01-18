#include "grayscale.hu"

int grayScaleSerial(char *pathIn, char *pathOut) {
    int width, height;
    int gray_value, r, g, b, i = 0;
    unsigned char *img_in = loadPPM(pathIn, &width, &height),
                  *img_out = (unsigned char *)malloc((width * height) * sizeof(unsigned char));

    if (img_out == nullptr) {
        handle_error("Errore nell'allocazione della memoria\n");
    }

    for (int y = 0; y < height; y += 1) {
        for (int x = 0; x < width; x += 1) {
            // prendi i valori di tre pixel contigui
            r = img_in[((y * width) + x) * 3];
            g = img_in[((y * width) + x)* 3 + 1];
            b = img_in[((y * width) + x)* 3 + 2];
            // Fai la media per prendere il grigio
            gray_value = (r + g + b) / CHANNELS;
            // Inseriscilo come primo pixel di img_out
            img_out[i++] = gray_value;
        }
    }

    writePPM(pathOut, img_out, width, height);

    return 0;
}


// Function to load the image from file
unsigned char* loadPPM(const char* path, int* width, int* height)
{
	FILE* file = fopen(path, "rb");

	if (!file) {
		fprintf(stderr, "Failed to open file\n");
		return NULL;
	}

	char header[3];
	fscanf(file, "%2s", header);
	if (header[0] != 'P' || header[1] != '6') {
		fprintf(stderr, "Invalid PPM file\n");
		return NULL;
	}

	fscanf(file, "%d %d", width, height);

	int maxColor;
	fscanf(file, "%d", &maxColor);

	fgetc(file);  // Skip single whitespace character

	unsigned char* img = (unsigned char*) malloc((*width) * (*height) * CHANNELS);
	if (!img) {
		fprintf(stderr, "Failed to allocate memory\n");
		return NULL;
	}

	fread(img, CHANNELS, *width * *height, file);

	fclose(file);

	return img;
}

// Function to write the matrix image to file
void writePPM(const char* path, unsigned char* img, int width, int height)
{
    FILE* file = fopen(path, "wb");

    if (!file) {
        fprintf(stderr, "Failed to open file\n");
        return;
    }

    fprintf(file, "P5\n%d %d\n255\n", width, height);

    fwrite(img, 3, width * height, file);

    fclose(file);
}