#include "main.hpp"


int main(int argc, char *argv[])
{

    uint width;
    uint height;
    unsigned char *img = loadPPM("/home/f3m/EdixProjects/matteo/immagine.ppm", &width, &height);

    printf("Numbero of max threads = %d\n", omp_get_max_threads());

    auto start = std::chrono::high_resolution_clock::now();

    blurSerial(img, (char *) "/home/f3m/Scrivania/Serial.ppm", width, height, 30);

    auto start2 = std::chrono::high_resolution_clock::now();

    blurOmp(img, (char *) "/home/f3m/Scrivania/Omp.ppm", width, height, 30);

    auto end = std::chrono::high_resolution_clock::now();

    auto deltaSerial = std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start);
    auto deltaOmp = std::chrono::duration_cast<std::chrono::milliseconds>(end - start2);

    printf(YELLOW BOLD "Serial" RESET " = %ld ms\n" YELLOW BOLD "Omp" RESET " = %ld ms\n", deltaSerial.count(), deltaOmp.count());



//    if (checkPostgresService() || checkRedisService())
//        exit(EXIT_FAILURE);
//    checkDb();
//    banner();
//    // TODO: magari puoi aprire direttamente un progetto passandolo come argomento
//    // TODO: ovunque si usi il path, mettere PATH_MAX oppure (meglio) far si che l'allocazione sia dinamica
//    // TODO: stiamo usando ovunque path assoluti, dovremmo usare dei path relativi
//    switch (argc)
//    {
//        case 1:
//            inputLoop();
//            break;
//        default:
//            fprintf(stderr, RED
//            "usage:"
//            RESET
//            " ./edix\n");
//            exit(EXIT_FAILURE);
//    }
//
//    //TODO: fare tutte le cose da fare prima di chiudere


    return 0;
}
int inputLoop()
{
    size_t lineSize = 256;
    char *line = (char *) malloc(256);

    size_t bytesRead;
    Env env = HOMEPAGE;
    bool stop = false;

    while (!stop && ((int) (bytesRead = getline(&line, &lineSize, stdin))) != -1)
    {
        if (bytesRead == 1) continue;

        line[bytesRead - 1] = '\0';
        switch (env)
        {
            case HOMEPAGE:
                parseHome(line, &env);
                break;
            case PROJECT:
                parseProj(line, &env);
                break;
            case SETTINGS:
                parseSettings(line, &env);
                break;
            case EXIT:
                //Unreachable
                break;
        }
        if (env == EXIT)
            stop = true;
    }

    free(line);
    D_PRINT("Uscita in corso...\n");

    return 0;
}

