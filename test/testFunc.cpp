#include "testFunc.hpp"

// Restituisce la percentuale di miglioramento del tempo t2 rispetto t1
double percentage(long time1, long time2) {
    return (((double)(time1 - time2) / time1) * 100.0);
}

void print_perc(double perc) {
    char *str = perc > 0.0 ? (char *)"Miglioramento" : (char *)"Peggioramento";
    printf("%s del " BOLD RED "%.2f\n" RESET, str, perc);
}

long print_times(auto time1, auto time2) {
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
    printf("Execution time: %ldms\n", time.count());
    return time.count();
}


void test(unsigned char *img1, unsigned char *img2, uint *width1, uint *height1, uint *width2, uint *height2) {
    uint outW, outH;
    unsigned char *outSer, *outParall;
    long timeSer = 0L,
         timePar = 0L;

    printf(BOLD YELLOW "Esecuzione seriale\n" RESET);
    auto startSer = std::chrono::high_resolution_clock::now();
    /* Scegli la funzione da chiamare Serial */

    //outSer = blurSerial(img1, *width1, *height1, 5, &outW, &outH);
    //outSer = grayscaleSerial(img1, *width1, *height1, &outW, &outH);
    //outSer = colorFilterSerial(img1, *width1, *height1, 0, 0, 255, 0, &outW, &outH);
    outSer = overlapOmp(img1, img2, *width1, *height1, *width2, *height2, 100, 200, &outW, &outH, 2);
    //outSer = compositionSerial(img1, img2, *width1, *height1, *width2, *height2, UP, &outW, &outH);
    //outSer = upscaleSerialBilinear(img1, *width1, *height1, 1, &outW, &outH);
    //outSer = upscaleSerialBicubic(img1, *width1, *height1, 1, &outW, &outH);
    //outSer = downscaleSerial(img1, *width1, *height1, 1, &outW, &outH);
    
    auto endSer = std::chrono::high_resolution_clock::now();
    timeSer = print_times(startSer, endSer);
    printf(BOLD YELLOW "%s" RESET, "\nEsecuzioni parallele\n\n");
    for (int i = 2; i <= 20; i++) {
        timePar = 0L;
        for (int j = 0; j < 100; j++) {
            auto start = std::chrono::high_resolution_clock::now();
            /* Scegli la funzione parallela (OMP, CUDA) */

            //outParall = blurOmp(img1, *width1, *height1, 5, &outW, &outH, i);
            //outParall = grayscaleOmp(img1, *width1, *height1, &outW, &outH, i);
            //outParall = colorFilterOmp(img1, *width1, *height1, 0, 0, 255, 0, &outW, &outH, i);
            outParall = overlapOmp(img1, img2, *width1, *height1, *width2, *height2, 20, 20, &outW, &outH, i);
            //outParall = compositionOmp(img1, img2, *width1, *height1, *width2, *height2, UP, &outW, &outH, i);
            //outParall = upscaleOmpBilinear(img1, *width1, *height1, 1, &outW, &outH, i);
            //outParall = upscaleOmpBicubic(img1, *width1, *height1, 1, &outW, &outH, i);
            //outParall = downscaleOmp(img1, *width1, *height1, 1, &outW, &outH, i); 

            auto end = std::chrono::high_resolution_clock::now();
            timePar += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
            double efficiency = ((double)timeSer / (i * timePar / 100)) * 100, 
                      speedup = (double)timeSer / (timePar / 100);

            printf("Con #thread=%d, tempo esecuzione medio= " BOLD GREEN "%.2f" RESET "\n", i, timePar/100.0);
            //print_perc(percentage(timeSer, timePar/100.0));
            printf("Efficiency: " BOLD BLUE "%.2f%\n" RESET, efficiency);
            printf("Speedup: " BOLD RED "%.2f\n" RESET, speedup);
            puts("");
    }
}

