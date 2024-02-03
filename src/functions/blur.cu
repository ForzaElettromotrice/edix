#include "blur.cuh"


int parseBlurArgs(char *args)
{
    char *imgIn = strtok(args, " ");
    char *pathOut = strtok(nullptr, " ");
    int radius = (int) strtol(strtok(nullptr, " "), nullptr, 10);

    if (imgIn == nullptr || pathOut == nullptr || radius <= 0)
    {
        handle_error("usage " BOLD "funx blur IN OUT RADIUS\n" RESET);
    }

    //TODO: leggere le immagini in base alla loro estensione
    char *tpp = getStrFromKey((char *) "TPP");
    uint width;
    uint height;
    uint channels;
    unsigned char *img = loadImage(imgIn, &width, &height, &channels);

    uint oWidth;
    uint oHeight;
    unsigned char *oImg;


    if (strcmp(tpp, "Serial") == 0)
        oImg = blurSerial(img, width, height, channels, radius, &oWidth, &oHeight);
    else if (strcmp(tpp, "OMP") == 0)
        oImg = blurOmp(img, width, height, channels, radius, &oWidth, &oHeight, 4);
    else if (strcmp(tpp, "CUDA") == 0)
        oImg = blurCuda(img, width, height, channels, radius, &oWidth, &oHeight);
    else
    {
        free(img);
        free(tpp);
        handle_error("Invalid arguments for blur function.\n");
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


unsigned char *blurSerial(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight)
{
    uint totalPixels = height * width;

    unsigned char *blurImage;

    blurImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * 3);
    if (blurImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore nell'allocare memoria\n");
        return nullptr;
    }

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            uint c[channels];
            memset(c, 0, channels * sizeof(uint));

            int num = 0;
            int curr_i;
            int curr_j;

            for (int m = -radius; m <= radius; m++)
            {
                for (int n = -radius; n <= radius; n++)
                {
                    curr_i = i + m;
                    curr_j = j + n;
                    if ((curr_i < 0) || (curr_i > width - 1) || (curr_j < 0) || (curr_j > height - 1)) continue;

                    for (int k = 0; k < channels; ++k)
                        c[k] += imgIn[(3 * (curr_i + curr_j * width)) + k];
                    num++;
                }
            }
            for (int k = 0; k < channels; ++k)
            {
                c[k] /= num;
                blurImage[3 * (i + j * width) + k] = c[k];
            }
        }
    }
    *oWidth = width;
    *oHeight = height;

    return blurImage;
}
unsigned char *blurOmp(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight, int nThread)
{
    uint totalPixels = height * width;

    auto *blurImage = (unsigned char *) malloc(sizeof(unsigned char *) * totalPixels * 3);
    if (blurImage == nullptr)
    {
        fprintf(stderr, RED "FUNX Error: " RESET "Errore nell'allocare memoria\n");
        return nullptr;
    }
//TODO: collapse
//TODO: schedule
//TODO: numero di thread ottimale
#pragma omp parallel for num_threads(nThread) default(none) shared(width, height, channels, radius, imgIn, blurImage)
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; i++)
        {
            if (j > height)
                continue;
            uint c[channels];
            memset(c, 0, channels * sizeof(uint));
            int num = 0;
            int curr_i;
            int curr_j;

            for (int m = -radius; m <= radius; m++)
            {
                for (int n = -radius; n <= radius; n++)
                {

                    curr_i = i + m;
                    curr_j = j + n;
                    if ((curr_i < 0) || (curr_i > width - 1) || (curr_j < 0) || (curr_j > height - 1)) continue;

                    for (int k = 0; k < channels; ++k)
                        c[k] += imgIn[(3 * (curr_i + curr_j * width)) + k];

                    num++;
                }
            }
            for (int k = 0; k < channels; ++k)
            {
                c[k] /= num;
                blurImage[3 * (i + j * width) + k] = c[k];
            }
        }
    }

    *oWidth = width;
    *oHeight = height;

    return blurImage;
}
unsigned char *blurCuda(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight)
{
    //host
    uint iSize = width * height * 3;
    auto h_blur_img = (unsigned char *) malloc(iSize * sizeof(unsigned char));
    mlock(imgIn, iSize * sizeof(unsigned char));

    //device
    unsigned char *d_img;
    unsigned char *d_blur_img;
    cudaMalloc(&d_img, iSize * sizeof(unsigned char));
    cudaMalloc(&d_blur_img, iSize * sizeof(unsigned char));

    //copy
    cudaMemcpy(d_img, imgIn, iSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //blur
    dim3 gridSize = {(width + 7) / 8, (height + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    size_t sharedDim = (size_t) pow(8 + 2 * radius, 2) * 3;
    blurShared<<<gridSize, blockSize, sharedDim>>>(d_img, d_blur_img, width, height, channels, radius);

    //copy back
    cudaMemcpy(h_blur_img, d_blur_img, iSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //free
    munlock(imgIn, iSize * sizeof(unsigned char));
    cudaFree(d_img);
    cudaFree(d_blur_img);

    *oWidth = width;
    *oHeight = height;
    return h_blur_img;
}
__global__ void blurShared(const unsigned char *img, unsigned char *blur_img, uint width, uint height, uint channels, int radius)
{
    int absX = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int absY = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (absX >= width || absY >= height)
        return;

    int relX = (int) threadIdx.x;
    int relY = (int) threadIdx.y;

    uint sDim = blockDim.x + 2 * radius;
    extern __shared__ unsigned char shared[];

    int x;
    int y;

    //Fa schifo, ma devo per forza allocare 3 e poi in caso uso solo 1
    int c[3];
    for (int k = 0; k < channels; ++k)
        c[k] = 0;

    int pixels = 0;

    if (relX == 0 && relY == 0)
    {

        for (int i = 0; i < sDim; ++i)
        {
            for (int j = 0; j < sDim; ++j)
            {
                x = absX - radius + i;
                y = absY - radius + j;
                if (x < 0 || x >= width || y < 0 || y >= height)
                    continue;

                for (int k = 0; k < channels; ++k)
                    shared[(i + j * sDim) * 3 + k] = img[(x + y * width) * 3 + k];
            }
        }
    }

    __syncthreads();

    for (int i = -radius; i <= radius; ++i)
    {
        for (int j = -radius; j <= radius; ++j)
        {
            x = relX + radius + i;
            y = relY + radius + j;

            if (absX + i < 0 || absX + i >= width || absY + j < 0 || absY + j >= height)
                continue;

            for (int k = 0; k < channels; ++k)
                c[k] += shared[(x + y * sDim) * 3 + k];
            pixels++;
        }
    }

    for (int k = 0; k < channels; ++k)
    {
        c[k] /= pixels;
        blur_img[(absX + absY * width) * 3 + k] = c[k];
    }
}