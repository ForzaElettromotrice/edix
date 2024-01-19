#include "functions.cuh"

int parseBlurArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int radius = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (imgIn == nullptr || pathOut == nullptr || radius == 0)
    {
        handle_error("Invalid arguments for blur function.\n");
    }

    //TODO: leggere le immagini in base alla loro estensione
    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;


    if (strcmp(tpp, "Serial") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        blurSerial(img, pathOut, width, height, radius);
        free(img);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        blurOmp(img, pathOut, width, height, radius);
        free(img);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        unsigned char *img = loadPPM(imgIn, &width, &height);
        blurCuda(img, pathOut, width, height, radius);
        free(img);
    } else
    {
        free(tpp);
        handle_error("Invalid arguments for blur function.\n");
    }
    free(tpp);

    return 0;
}


int blurSerial(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius){
	int totalPixels=height * width;

    unsigned char * blurImage;

	blurImage = (unsigned char *)malloc(sizeof(unsigned char*) * totalPixels * CHANNELS);
	if(blurImage == nullptr){
		fprintf(stderr, RED "FUNX Error: " RESET "Errore nell'allocare memoria\n");
		return 1;
	}

	for(int i = 0; i < width; i++) {
		for(int j = 0; j < height; j++) {

			unsigned int red  =0;
		    unsigned int green=0;
		    unsigned int blue =0;

		    int num=0;
		    int curr_i, curr_j;
			//borders of every px
			for (int m = -radius; m <= radius; m++) {
				for (int n = -radius; n <= radius; n++) {

					curr_i = i + m;
					curr_j = j + n;
					//check if iteration is out of borders
					if((curr_i<0)||(curr_i>height-1)||(curr_j<0)||(curr_j>width-1)) continue;

					//to access red channel column +(row*columns)
					red   += imgIn[(3*(curr_j+curr_i*width))];
					//to access green channel column +(row*columns) + 1
					green += imgIn[(3*(curr_j+curr_i*width))+1];
					//to access blue channel column +(row*columns) + 2
					blue  += imgIn[(3*(curr_j+curr_i*width))+2];

					num++;
				}
			}

			red /= num;
			green /= num;
			blue /= num;

			blurImage[3*(j+i*width)] = red;
		    blurImage[3*(j+i*width)+1] = green;
		    blurImage[3*(j+i*width)+2] = blue;

		}
	}

    char *tmp;

	tmp = (char*) malloc((strlen(pathOut)+10) * sizeof(char));
	if(tmp == nullptr){
		fprintf(stderr, RED "FUNX Error: " RESET "Errore nell'allocare memoria\n");
		return 1;
	}
	strcpy(tmp, pathOut);
    writePPM(strcat(tmp,".ppm"), blurImage, width, height,"P6");

    free(blurImage);
    free(tmp);

	return 0;
}

int blurOmp(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius)
{
    return 0;
}

int blurCuda(unsigned char *imgIn, char *pathOut, uint width, uint height, int radius)
{
    return 0;
}
