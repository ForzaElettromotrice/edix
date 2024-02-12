#include "blur.cuh"

__global__ void blur(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, uint channels, int radius)
{
    int x = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= width || y >= height)
        return;

    int RGB[3];
    for (int k = 0; k < channels; ++k)
        RGB[k] = 0;

    int curr_i;
    int curr_j;
    int num = 0;

    for (int i = -radius; i <= radius; ++i)
    {
        for (int j = -radius; j <= radius; ++j)
        {
            curr_i = x + i;
            curr_j = y + j;
            if (curr_i < 0 || curr_j < 0 || curr_i >= width || curr_j >= height)
                continue;

            for (int k = 0; k < channels; ++k)
                RGB[k] += imgIn[(curr_i + curr_j * width) * channels + k];
            num++;
        }
    }

    for (int k = 0; k < channels; ++k)
    {
        RGB[k] /= num;
        imgOut[(x + y * width) * channels + k] = RGB[k];
    }


}
__global__ void blurShared(const unsigned char *imgIn, unsigned char *imgOut, uint width, uint height, uint channels, int radius)
{
    int absX = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int absY = (int) (threadIdx.y + blockIdx.y * blockDim.y);

    if (absX >= width || absY >= height)
        return;

    int relX = (int) threadIdx.x;
    int relY = (int) threadIdx.y;

    int sSide = (int) blockDim.x + 2 * radius;
    extern __shared__ unsigned char shared[];


    int amount = (int) pow(((sSide + blockDim.x - 1) / blockDim.x), 2);
    int jump = (int) (blockDim.x * blockDim.y);

    for (int n = 0; n < amount; ++n)
    {
        int index = relX + relY * (int) blockDim.x + n * jump;
        int nRelX = index % sSide;
        int nRelY = index / sSide;
        int nAbsX = nRelX - radius + (int) (blockIdx.x * blockDim.x);
        int nAbsY = nRelY - radius + (int) (blockIdx.y * blockDim.y);

        if (index >= pow(sSide, 2))
            break;

        if (nAbsX >= width || nAbsY >= height || nAbsX < 0 || nAbsY < 0)
            continue;

        for (int k = 0; k < channels; ++k)
            shared[index * channels + k] = imgIn[(nAbsX + nAbsY * width) * channels + k];

    }
    __syncthreads();

    int RGB[3];
    for (int k = 0; k < channels; ++k)
        RGB[k] = 0;

    int x;
    int y;
    int num = 0;

    for (int i = -radius; i <= radius; ++i)
        for (int j = -radius; j <= radius; ++j)
        {
            x = relX + radius + i;
            y = relY + radius + j;

            if (absX + i < 0 || absX + i >= width || absY + j < 0 || absY + j >= height)
                continue;

            for (int k = 0; k < channels; ++k)
                RGB[k] += shared[(x + y * sSide) * channels + k];
            num++;
        }

    for (int k = 0; k < channels; ++k)
    {
        RGB[k] /= num;
        imgOut[(absX + absY * width) * channels + k] = RGB[k];
    }
}


unsigned char *blurSerial(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight)
{
    *oWidth = width;
    *oHeight = height;

    uint oSize = width * height * channels;
    auto imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }

    int RGB[channels];
    memset(RGB, 0, channels * sizeof(int));

    int num;
    int curr_i;
    int curr_j;

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            num = 0;

            for (int i = -radius; i <= radius; i++)
                for (int j = -radius; j <= radius; j++)
                {
                    curr_i = x + i;
                    curr_j = y + j;
                    if ((curr_i < 0) || (curr_i >= width) || (curr_j < 0) || (curr_j >= height))
                        continue;

                    for (int k = 0; k < channels; ++k)
                        RGB[k] += imgIn[(curr_i + curr_j * width) * channels + k];
                    num++;
                }
            for (int k = 0; k < channels; ++k)
            {
                RGB[k] /= num;
                imgOut[(x + y * width) * channels + k] = RGB[k];
            }
        }


    return imgOut;
}
unsigned char *blurOmp(const unsigned char *imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight, int nThreads)
{
    *oWidth = width;
    *oHeight = height;

    uint oSize = width * height * channels;
    auto imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }

    int RGB[channels];
    memset(RGB, 0, channels * sizeof(int));

    int num;
    int curr_i;
    int curr_j;

//TODO: magari cambiare schedule
#pragma omp parallel for num_threads(nThreads) collapse(2) schedule(static) default(none) shared(width, height, channels, imgIn, imgOut, radius) private(RGB, num, curr_i, curr_j)
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            num = 0;
            for (int i = -radius; i <= radius; i++)
                for (int j = -radius; j <= radius; j++)
                {
                    curr_i = x + i;
                    curr_j = y + j;
                    if ((curr_i < 0) || (curr_i >= width) || (curr_j < 0) || (curr_j >= height))
                        continue;

                    for (int k = 0; k < channels; ++k)
                        RGB[k] += imgIn[(curr_i + curr_j * width) * channels + k];
                    num++;
                }
            for (int k = 0; k < channels; ++k)
            {
                RGB[k] /= num;
                imgOut[(x + y * width) * channels + k] = RGB[k];
            }
        }


    return imgOut;
}
unsigned char *blurCuda(const unsigned char *h_imgIn, uint width, uint height, uint channels, int radius, uint *oWidth, uint *oHeight, bool useShared)
{
    *oWidth = width;
    *oHeight = height;

    uint oSize = width * height * channels;
    auto h_imgOut = (unsigned char *) malloc(oSize * sizeof(unsigned char));
    if (h_imgOut == nullptr)
    {
        E_Print("Errore durante la malloc!\n");
        return nullptr;
    }
    mlock(h_imgIn, oSize * sizeof(unsigned char));


    unsigned char *d_imgIn;
    unsigned char *d_imgOut;
    cudaMalloc(&d_imgIn, oSize * sizeof(unsigned char));
    cudaMalloc(&d_imgOut, oSize * sizeof(unsigned char));

    cudaMemcpy(d_imgIn, h_imgIn, oSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 gridSize = {(width + 7) / 8, (height + 7) / 8, 1};
    dim3 blockSize = {8, 8, 1};
    if (useShared)
    {
        auto sharedDim = (size_t) pow(8 + 2 * radius, 2) * channels;
        blurShared<<<gridSize, blockSize, sharedDim>>>(d_imgIn, d_imgOut, width, height, channels, radius);
    } else
        blur<<<gridSize, blockSize>>>(d_imgIn, d_imgOut, width, height, channels, radius);

    cudaMemcpy(h_imgOut, d_imgOut, oSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    munlock(h_imgIn, oSize * sizeof(unsigned char));
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);

    return h_imgOut;
}