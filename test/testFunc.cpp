#include "testFunc.hpp"

// Restituisce la percentuale di miglioramento del tempo t2 rispetto t1
double percentage(long time1, long time2) {
    return (((double)(time1 - time2) / time1) * 100.0);
}

void print_perc(double perc) {
    char *str = perc > 0.0 ? (char *)"Miglioramento" : (char *)"Peggioramento";
    printf("%s del " BOLD RED "%.2f\n" RESET, str);
}

long print_times(auto time1, auto time2) {
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1);
    printf("Execution time: %ldus\n", time.count());
    return time.count();
}


void test(unsigned char *img1, unsigned char *img2, uint *width1, uint *height1, uint *width2, uint *height2) {
    uint outW, outH;
    unsigned char *outSer, *outParall;
    long timePar = 0L;

    printf(BOLD YELLOW "Esecuzione seriale\n" RESET);
    auto startSer = std::chrono::high_resolution_clock::now();
    // Scegli la funzione da chiamare Serial
    outSer = compositionSerial(img1, img2, *width1, *height1, *width2, *height2, UP, &outW, &outH);
    auto endSer = std::chrono::high_resolution_clock::now();
    long timeSer = print_times(startSer, endSer);
    printf(BOLD YELLOW "%s" RESET, "\nEsecuzioni parallele\n\n");
    for (int i = 2; i <= 20; i++) {
        for (int j = 0; j < 100; j++) {
            auto start = std::chrono::high_resolution_clock::now();
            outParall = compositionOmp(img1, img2, *width1, *height1, *width2, *height2, UP, &outW, &outH, i);
            auto end = std::chrono::high_resolution_clock::now();
            timePar += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
            printf("Con #thread=%d, tempo esecuzione medio= " BOLD GREEN "%.2f" RESET "\n", i, timePar/100.0);
            print_perc(percentage(timeSer, timePar/100.0));
            puts("");
    }
}

