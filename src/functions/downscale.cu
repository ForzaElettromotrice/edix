//
// Created by f3m on 19/01/24.
//

#include "downscale.cuh"

__global__ void bilinearDownscaleCUDA(const unsigned char *imgIn, unsigned char *imgOut,uint width, uint height, int factor)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    uint widthO = width / factor;
    uint heightO = height / factor;

    uint idx = x + y * widthO;

    int i;
    int j;
    int p00;
    int p01;
    int p10;
    int p11;
    double alpha;
    double beta;

    if (idx < widthO * heightO * 3){
        i = x * factor;
        j = y * factor;
        alpha = ((double) x * factor)-i;
        beta = ((double) y * factor)-j;
        
        for(int k = 0; k < 3 ; k++){
            p00 = imgIn[(i + j * width)*3 + k];
            p01 = imgIn[(i + 1 + j * width)*3 + k];
            p10 = imgIn[(i + (j + 1) * width)*3 + k];
            p11 = imgIn[(i + 1 + (j + 1) * width)*3 + k];

            imgOut[(idx * 3) +k] = (int) ((1 - alpha) * (1 - beta) * p00 + (1 - alpha) *
                                                 beta * p01 + alpha * (1 - beta) * p10 +
                                                  alpha * beta * p11);
        }
    } 
}

int parseDownscaleArgs(char *args)
{
    char *pathIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int factor = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (pathIn == nullptr || pathOut == nullptr || factor == 0)
    {
        handle_error("Invalid arguments for upscale\n");
    }

    char *tpp = getStrFromKey((char *) "TPP");
    char *tup = getStrFromKey((char *) "TUP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(pathIn, &width, &height, &channels);

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;

    if (strcmp(tpp, "Serial") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = downscaleSerialBilinear(img, width, height, channels, factor, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = downscaleSerialBicubic(img, width, height, channels, factor, &oWidth, &oHeight);
    } else if (strcmp(tpp, "OMP") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = downscaleOmpBilinear(img, width, height, channels, factor, &oWidth, &oHeight, 4);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = downscaleOmpBicubic(img, width, height, channels, factor, &oWidth, &oHeight, 4);
    } else if (strcmp(tpp, "CUDA") == 0)
    {
        if (strcmp(tup, "Bilinear") == 0)
            oImg = downscaleCudaBilinear(img, width, height, channels, factor, &oWidth, &oHeight);
        else if (strcmp(tup, "Bicubic") == 0)
            oImg = downscaleCudaBicubic(img, width, height, channels, factor, &oWidth, &oHeight);
    } else
    {
        free(tpp);
        handle_error("Invalid TPP\n");
    }

    if (oImg != nullptr)
    {
        writeImage(pathOut, oImg, oWidth, oHeight, channels);
        free(oImg);
    }

    free(img);
    free(tpp);
    return 0;
}

unsigned char *downscaleSerialBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{

    uint widthO = width / factor;
    uint heightO = height / factor;
    auto *imgOut = (unsigned char *) malloc((widthO * heightO * 3) * sizeof(unsigned char));

    int x;
    int y;
    int p00;
    int p01;
    int p10;
    int p11;
    double alpha;
    double beta;

    for (int i = 0; i < widthO; ++i)
    {
        for (int j = 0; j < heightO; ++j)
        {
            x = i * factor;
            y = j * factor;
            alpha = ((double) i * factor) - x + 0.5;
            beta = ((double) j * factor) - y + 0.5;


            for (int k = 0; k < channels; ++k)
            {
                //TODO: se sbordi, usa lo stesso pixel
                p00 = imgIn[(x + y * width) * 3 + k];
                p01 = imgIn[(x + 1 + y * width) * 3 + k];
                p10 = imgIn[(x + (y + 1) * width) * 3 + k];
                p11 = imgIn[(x + 1 + (y + 1) * width) * 3 + k];


                imgOut[(i + j * widthO) * 3 + k] = bilinearInterpolation(p00, p01, p10, p11, alpha, beta);
            }
        }
    }


    *oWidth = widthO;
    *oHeight = heightO;

    return imgOut;
}
unsigned char *downscaleOmpBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread)
{
    return nullptr;
}
unsigned char *downscaleCudaBilinear(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{
    return nullptr;
}

unsigned char *downscaleSerialBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{
    return nullptr;
}
unsigned char *downscaleOmpBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight, int nThread)
{
    return nullptr;
}
unsigned char *downscaleCudaBicubic(const unsigned char *imgIn, uint width, uint height, uint channels, int factor, uint *oWidth, uint *oHeight)
{
    //init
    uint wf,hf;

    //host
    wf = width / factor;
    hf = height / factor;
    uint iSize = width * height * 3;
    uint iSizeO = wf * hf * 3;
    auto h_imgOut = (unsigned char *) malloc(iSizeO * sizeof(unsigned char));
    if (h_imgOut == nullptr){
        fprintf(stderr, RED "Error: " RESET "Errore nell'allocazione della memoria\n");
        munlock(imgIn,iSize * sizeof(unsigned char));
        return nullptr;
    }
    mlock(imgIn, iSize * sizeof(unsigned char));

    //device
    unsigned char *d_imgIn;
    unsigned char *d_imgOut;
    cudaMalloc(&d_imgIn, iSize * sizeof(unsigned char));
    cudaMalloc(&d_imgOut, iSizeO * sizeof(unsigned char));

    //copy
    cudaMemcpy(d_imgIn, imgIn, iSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //upscale
    dim3 gridSize = {(wf + 7) / 8, (hf + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    bilinearDownscaleCUDA<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, height, factor);

    //copy back
    cudaMemcpy(h_imgOut, d_imgOut, iSizeO * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //free
    munlock(imgIn, iSize * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    *oWidth = wf;
    *oHeight = hf;

    return h_imgOut;
}