#include "grayscale.cuh"


__global__ void grayscaleCUDA(unsigned char *ImgIn, unsigned char *ImgOut, uint width, uint height){
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height){

        unsigned char r = ImgIn[(y * width + x)*3];
        unsigned char g = ImgIn[(y * width + x)*3+1];
        unsigned char b = ImgIn[(y * width + x)*3+2];

        ImgOut[(y * width + x)] = (r + g + b) / 3;
    }
    
}

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
    unsigned char *img;
    char format[3];

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
    {
        img = loadPPM(imgIn, &width, &height, format);
        oImg = grayscaleSerial(img, width, height, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        img = loadPPM(imgIn, &width, &height, format);
        oImg = grayscaleOmp(img, width, height, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        img = loadPPM(imgIn, &width, &height, format);
        oImg = grayscaleCuda(img, width, height, &oWidth, &oHeight);
    } else
    {
        free(tpp);
        handle_error("Errore nel parsing degli argomenti\n");
    }
    if (oImg != nullptr)
    {
        //TODO: scrivere le immagini in base all'estenzione
        writePPM(pathOut, oImg, oWidth, oHeight, "P5");
        free(oImg);
    }
    free(img);
    free(tpp);
    return 0;
}

unsigned char *grayscaleSerial(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight)
{
    auto *img_out = (unsigned char *) malloc((width * height) * sizeof(unsigned char));

    if (img_out == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Errore nell'allocazione della memoria\n");
        return nullptr;
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
            grayValue = (r + g + b) / 3;
            // Inseriscilo come primo pixel di img_out
            img_out[i++] = grayValue;
        }
    }

    *oWidth = width;
    *oHeight = height;

    return img_out;
}
unsigned char *grayscaleOmp(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight, int nThread)
{
    auto *img_out = (unsigned char *) malloc((width * height) * sizeof(unsigned char));

    if (img_out == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Errore nell'allocazione della memoria");
        return nullptr;
    }

    int r, g, b, grayValue;

    //TODO: schedule e controllo su num_threads
#pragma omp parallel for num_threads(nThread) default(none) private(r, g, b, grayValue) shared(img_out, imgIn, width, height)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            r = imgIn[((y * width) + x) * 3];
            g = imgIn[((y * width) + x) * 3 + 1];
            b = imgIn[((y * width) + x) * 3 + 2];
            grayValue = (r + g + b) / 3;

            img_out[y * width + x] = grayValue;
        }
    }

    *oWidth = width;
    *oHeight = height;

    return img_out;
}
unsigned char *grayscaleCuda(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight)
{
    unsigned char *h_imgOut;

    unsigned char *d_imgOut;
    unsigned char *d_imgIn;
    

    mlock(imgIn,width * height * 3 * sizeof(unsigned char));
    h_imgOut = (unsigned char *)malloc(width * height * sizeof(unsigned char*));
    if (h_imgOut == nullptr){
        fprintf(stderr, RED "Error: " RESET "Errore nell'allocazione della memoria\n");
        munlock(imgIn, width * height * 3 * sizeof(unsigned char));
        return nullptr;
    }

    cudaMalloc((void **) &d_imgIn,width * height * 3 * sizeof(unsigned char));
    cudaMemcpy(d_imgIn,imgIn,width * height * 3 * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_imgOut,width * height * sizeof(unsigned char));

    dim3 gridSize((width+7)/8 , (height+7)/8);
    dim3 blockSize(8,8,1);

    grayscaleCUDA<<<gridSize,blockSize>>>(d_imgIn,d_imgOut,width,height);

    cudaMemcpy(h_imgOut, d_imgOut, width * height, cudaMemcpyDeviceToHost);

    munlock(imgIn,width * height * 3 * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    *oWidth = width;
    *oHeight = height;

    return h_imgOut;
}


