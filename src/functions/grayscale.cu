#include "grayscale.cuh"


__global__ void grayscale(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height)
{
    int x = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= width || y >= height)
        return;

    double RGB[] = {0, 0, 0};

    for (int k = 0; k < 3; ++k)
        RGB[k] = imgIn[(x + y * width) * 3 + k];

    imgOut[x + y * width] = (int) (RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722);
}


unsigned char *grayscaleSerial(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight)
{
    *oWidth = width;
    *oHeight = height;

    uint oSize = width * height;
    auto imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }


    double RGB[] = {0, 0, 0};

    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
        {
            for (int k = 0; k < 3; ++k)
                RGB[k] = imgIn[(i + j * width) * 3 + k];

            imgOut[i + j * width] = (int) (RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722);
        }


    return imgOut;
}
unsigned char *grayscaleOmp(const unsigned char *imgIn, uint width, uint height, uint *oWidth, uint *oHeight, int nThreads)
{
    *oWidth = width;
    *oHeight = height;

    uint oSize = width * height;
    auto imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }


    double RGB[] = {0, 0, 0};

#pragma omp parallel for num_threads(nThreads) collapse(2) schedule(static) default(none) shared(width, height, imgIn, imgOut) private(RGB)
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
        {
            for (int k = 0; k < 3; ++k)
                RGB[k] = imgIn[(i + j * width) * 3 + k];

            imgOut[i + j * width] = (int) (RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722);
        }


    return imgOut;
}
unsigned char *grayscaleCuda(const unsigned char *h_imgIn, uint width, uint height, uint *oWidth, uint *oHeight)
{
    *oWidth = width;
    *oHeight = height;

    uint oSize = width * height;
    uint iSize = width * height * 3;

    auto h_imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (h_imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }
    mlock(h_imgIn, oSize * sizeof(unsigned char));


    unsigned char *d_imgIn;
    unsigned char *d_imgOut;
    cudaMalloc(&d_imgIn, iSize * sizeof(unsigned char));
    cudaMalloc(&d_imgOut, oSize * sizeof(unsigned char));

    cudaMemcpy(d_imgIn, h_imgIn, iSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 gridSize = {(width + 7) / 8, (height + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    grayscale<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, height);

    cudaMemcpy(h_imgOut, d_imgOut, oSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    munlock(h_imgIn, oSize * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    return h_imgOut;
}