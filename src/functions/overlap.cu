//
// Created by f3m on 19/01/24.
//

#include "overlap.cuh"

int parseOverlapArgs(char *args)
{
    char *img1 = strtok(args, " ");
    char *img2 = strtok(nullptr, " ");
    char *pathOut = strtok(nullptr, " ");
    //TODO: controllare se x e y sono stringhe o meno
    uint x = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);
    uint y = (uint) strtoul(strtok(nullptr, " "), nullptr, 10);

    if (img1 == nullptr || img2 == nullptr || pathOut == nullptr)
    {
        handle_error("Invalid arguments for overlap\n");
    }

    char *tpp = getStrFromKey((char *) "TPP");
    uint width1;
    uint height1;
    uint channels1;
    uint width2;
    uint height2;
    uint channels2;
    unsigned char *img1_1 = loadImage(img1, &width1, &height1, &channels1);
    unsigned char *img2_1 = loadImage(img2, &width2, &height2, &channels2);

    uint oWidth;
    uint oHeight;
    uint oChannels = channels1 == 3 ? 3 : channels2;
    unsigned char *oImg;


    if (strcmp(tpp, "Serial") == 0)
        oImg = overlapSerial(img1_1, img2_1, width1, height1, channels1, width2, height2, channels2, x, y, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = overlapOmp(img1_1, img2_1, width1, height1, channels1, width2, height2, channels2, x, y, &oWidth, &oHeight, 4);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = overlapCuda(img1_1, img2_1, width1, height1, channels1, width2, height2, channels2, x, y, &oWidth, &oHeight);
    else
    {
        free(img1_1);
        free(img2_1);
        free(tpp);
        handle_error("Invalid TPP\n");
    }

    if (oImg != nullptr)
    {
        writeImage(pathOut, oImg, oWidth, oHeight, oChannels);
        free(oImg);
    }

    free(img1_1);
    free(img2_1);
    free(tpp);
    return 0;
}

unsigned char *overlapSerial(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight)
{
    if (x + width2 > width1 || y + height2 > height1)
    {
        fprintf(stderr, RED "Error: " RESET "La secondo immagine è troppo grande per essere inserita li!\n");
        return nullptr;
    }

    uint oSize = width1 * height1 * (channels2 == 3 ? 3 : channels1);
    auto *oImg = (unsigned char *) malloc(oSize * sizeof(unsigned char));

    if (channels1 == 3 || channels1 == channels2)
        memcpy(oImg, img1, oSize * sizeof(unsigned char));
    else
    {
        for (int i = 0; i < width1; i++)
            for (int j = 0; j < height1; j++)
                for (int k = 0; k < channels2; ++k)
                    oImg[(i + j * width1) * channels2 + k] = img1[i + j * width1];
    }
    for (int i = 0; i < width2; i++)
        for (int j = 0; j < height2; j++)
        {
            if (channels2 == 3 || channels2 == channels1)
            {
                for (int k = 0; k < channels2; ++k)
                    oImg[3 * (x + i + (y + j) * width1) + k] = img2[3 * (i + j * width2) + k];
            } else
            {
                for (int k = 0; k < channels2; ++k)
                    oImg[3 * (x + i + (y + j) * width1) + k] = img2[i + j * width2];
            }
        }

    *oWidth = width1;
    *oHeight = height1;


    return oImg;
}
unsigned char *overlapOmp(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight, int nThread)
{
    return nullptr;
}
unsigned char *overlapCuda(const unsigned char *img1, const unsigned char *img2, uint width1, uint height1, uint channels1, uint width2, uint height2, uint channels2, uint x, uint y, uint *oWidth, uint *oHeight)
{
    if (x + width2 > width1 || y + height2 > height1)
    {
        fprintf(stderr, RED "Error: " RESET "La secondo immagine è troppo grande per essere inserita li!\n");
        return nullptr;
    }
    //host
    uint iSize = width1 * height1 * 3;
    auto h_oImg = (unsigned char *) malloc(iSize * sizeof(unsigned char));
    mlock(img1, iSize * sizeof(unsigned char));
    memcpy(h_oImg, img1, iSize * sizeof(unsigned char));

    //device
    unsigned char *d_oImg;
    unsigned char *d_img2;
    cudaMalloc(&d_oImg, iSize * sizeof(unsigned char));
    cudaMalloc(&d_img2, width2 * height2 * 3 * sizeof(unsigned char));

    //copy
    cudaMemcpy(d_oImg, h_oImg, iSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2, width2 * height2 * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //overlap
    dim3 gridSize = {(width2 + 7) / 8, (height2 + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    overlap<<<gridSize, blockSize>>>(d_oImg, d_img2, width1, width2, height2, x, y);

    //copy back
    cudaMemcpy(h_oImg, d_oImg, iSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //free
    munlock(h_oImg, iSize * sizeof(unsigned char));
    cudaFree(d_oImg);
    cudaFree(d_img2);


    *oWidth = width1;
    *oHeight = height1;
    return h_oImg;
}

__global__ void overlap(unsigned char *img, const unsigned char *img2, uint width, uint width2, uint height2, uint posX, uint posY)
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width2 || y >= height2)
        return;

    img[(x + posX + (y + posY) * width) * 3] = img2[(x + y * width2) * 3];
    img[(x + posX + (y + posY) * width) * 3 + 1] = img2[(x + y * width2) * 3 + 1];
    img[(x + posX + (y + posY) * width) * 3 + 2] = img2[(x + y * width2) * 3 + 2];
}