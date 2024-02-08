#include "grayscale.cuh"


__global__ void grayscaleCUDA(const unsigned char *ImgIn, unsigned char *ImgOut, uint width, uint height)
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {

        unsigned char r = ImgIn[(y * width + x) * 3];
        unsigned char g = ImgIn[(y * width + x) * 3 + 1];
        unsigned char b = ImgIn[(y * width + x) * 3 + 2];

        ImgOut[(y * width + x)] = (r + g + b) / 3;
    }

}

unsigned char *grayscaleSerial(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight)
{
    auto *img_out = (unsigned char *) malloc((width * height) * sizeof(unsigned char));

    if (img_out == nullptr)
    {
        E_Print(RED
                "Error: "
                RESET
                "Errore nell'allocazione della memoria\n");
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
            r = imgIn[((y * width) + x) * 3];
            g = imgIn[((y * width) + x) * 3 + 1];
            b = imgIn[((y * width) + x) * 3 + 2];

            grayValue = (r + g + b) / 3;

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
        E_Print(RED "Error: " RESET "Errore nell'allocazione della memoria");
        return nullptr;
    }

    int r, g, b, grayValue, index;
#pragma omp parallel for num_threads(nThread) collapse(2) default(none) private(r, g, b, grayValue, index) shared(img_out, imgIn, width, height) schedule(static)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            index = y * width + x;
            r = imgIn[index * 3];
            g = imgIn[index * 3 + 1];
            b = imgIn[index * 3 + 2];
            grayValue = (r + g + b) / 3;

            img_out[index] = grayValue;
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


    mlock(imgIn, width * height * 3 * sizeof(unsigned char));
    h_imgOut = (unsigned char *) malloc(width * height * sizeof(unsigned char *));
    if (h_imgOut == nullptr)
    {
        E_Print(RED "Error: " RESET "Errore nell'allocazione della memoria\n");
        munlock(imgIn, width * height * 3 * sizeof(unsigned char));
        return nullptr;
    }

    cudaMalloc((void **) &d_imgIn, width * height * 3 * sizeof(unsigned char));
    cudaMemcpy(d_imgIn, imgIn, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_imgOut, width * height * sizeof(unsigned char));

    dim3 gridSize((width + 7) / 8, (height + 7) / 8);
    dim3 blockSize(8, 8, 1);

    grayscaleCUDA<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, height);

    cudaMemcpy(h_imgOut, d_imgOut, width * height, cudaMemcpyDeviceToHost);

    munlock(imgIn, width * height * 3 * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    *oWidth = width;
    *oHeight = height;

    return h_imgOut;
}


