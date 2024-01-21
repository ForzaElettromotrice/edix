#include "main.hpp"


int main(int argc, char *argv[])
{

    uint width = 51;
    uint height = 1;
    unsigned char *img = loadPPM("/home/f3m/EdixProjects/matteo/tmp.ppm", &width, &height);
//    auto *img = (unsigned char *) malloc(68 * sizeof(unsigned char));
//    sprintf((char *) img, "vaaaaaaaaaaaafahofuahofaofiahfoiahfoaihfaiofahfoahf");
    printf(BOLD BLUE "img " RESET "= " YELLOW ITALIC "%d Byte\n" RESET, width * height * 3);
//    printf(BOLD BLUE "img " RESET "= " YELLOW ITALIC "%d Byte\n" RESET, width * height);
    printf(ITALIC "Applico algoritmo lzTransformer...\n" RESET);
//    unsigned char *cImg;
//    for (int i = 1; i < 100; ++i)
//    {
//        size_t cSize;
//        cImg = lzTransformer(img, width * height * 3, &cSize, i);
//        printf(BOLD BLUE "Compressed img" RESET " = " GREEN ITALIC "%zu Byte\t" RESET BOLD RED "i = %d\n" RESET, cSize, i);
//    }

    size_t cSize;
    unsigned char *cImg;
    cImg = lzTransformer(img, width * height * 3, &cSize, 9);
//    cImg = lzTransformer(img, width * height, &cSize, 9);
    printf(BOLD BLUE "Compressed img" RESET " = " GREEN ITALIC "%zu Byte\t" RESET BOLD RED "i = %d\n" RESET, cSize, 13);


    size_t oSize;
    unsigned char *oImg;
    oImg = decoder(cImg, cSize, &oSize);

    printf(BOLD BLUE "Decompressed img" RESET " = " GREEN ITALIC "%zu Byte\n" RESET, oSize);

//    printf("%s\n%s\n", img, oImg);

    writePPM("/home/f3m/Scrivania/out.ppm", oImg, width, height, "P6");

    free(img);
    free(cImg);
    free(oImg);

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
//            fprintf(stderr, RED "usage:" RESET " ./edix\n");
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

