#include "testFunc.hpp"

// Restituisce la percentuale di miglioramento del tempo t2 rispetto t1
double percentage(long time1, long time2)
{
    return (((double) (time1 - time2) / (double) time1) * 100.0);
}

void print_perc(double perc)
{
    char *str = perc > 0.0 ? (char *) "Miglioramento" : (char *) "Peggioramento";
    printf("%s del " BOLD RED "%.2f\n" RESET, str, perc);
}

long print_times(auto time1, auto time2)
{
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
    printf("Execution time: %ldms\n", time.count());
    return time.count();
}


void test(const unsigned char *img1, const unsigned char *img2, const uint *width1, const uint *height1, const uint *width2, const uint *height2, const uint *channels1, const uint *channels2)
{
    uint outW, outH;
    long timeSer, timePar;

    printf(BOLD YELLOW "Esecuzione seriale\n" RESET);
    auto startSer = std::chrono::high_resolution_clock::now();
    /* Scegli la funzione da chiamare Serial */

    //free(blurSerial(img1, *width1, *height1, 5, &outW, &outH)); 
    //free(grayscaleSerial(img1, *width1, *height1, &outW, &outH));
    //free(colorFilterSerial(img1, *width1, *height1, 0, 0, 255, 0, &outW, &outH));
    //free(overlapSerial(img1, img2, *width1, *height1, *width2, *height2, 100, 200, &outW, &outH));
    //free(compositionSerial(img1, img2, *width1, *height1, *width2, *height2, UP, &outW, &outH));
    free(upscaleSerialBilinear(img1, *width1, *height1, 3, 1, &outW, &outH));
    //free(upscaleSerialBicubic(img1, *width1, *height1, 1, &outW, &outH));
    //free(downscaleSerialBilinear(img1, *width1, *height1, 1, &outW, &outH));

    auto endSer = std::chrono::high_resolution_clock::now();
    timeSer = print_times(startSer, endSer);
    printf(BOLD YELLOW "%s" RESET, "\nEsecuzioni parallele\n");
    for (int i = 2; i <= 20; i++)
    {
        timePar = 0L;
        for (int j = 0; j < 100; j++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            /* Scegli la funzione parallela (OMP, CUDA) */

            //free(blurOmp(img1, *width1, *height1, 5, &outW, &outH, i));
            //free(grayscaleOmp(img1, *width1, *height1, &outW, &outH, i));
            //free(colorFilterOmp(img1, *width1, *height1, 0, 0, 255, 0, &outW, &outH, i));
            //free(overlapOmp(img1, img2, *width1, *height1, *width2, *height2, 20, 20, &outW, &outH, i));
            //free(compositionOmp(img1, img2, *width1, *height1, *width2, *height2, UP, &outW, &outH, i));
            free(upscaleOmpBilinear(img1, *width1, *height1, 3, 1, &outW, &outH, i));
            //free(upscaleOmpBicubic(img1, *width1, *height1, 1, &outW, &outH, i));
            //free(downscaleOmpBilinear(img1, *width1, *height1, 1, &outW, &outH, i));

            auto end = std::chrono::high_resolution_clock::now();
            timePar += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
        double efficiency = ((double) timeSer / (i * (double) timePar / 100)) * 100,
                speedup = (double) timeSer / ((double) timePar / 100);

        printf(BOLD "threads: " RESET BLUE "%2d " RESET BOLD "time: " RESET GREEN "%.2f " RESET, i, (double) timePar / 100);
        printf(BOLD "S: " RESET RED "%.2f " RESET, speedup);
        printf(BOLD "E: " RESET BLUE "%.2f%s\n" RESET, efficiency, "%");
    }
}

