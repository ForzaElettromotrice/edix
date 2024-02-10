#include "main.hpp"


int main(int argc, char *argv[])
{
    if (checkPostgresService() || checkRedisService())
        exit(EXIT_FAILURE);
    checkDb();
    banner();
    // TODO: magari puoi aprire direttamente un progetto passandolo come argomento
    // TODO: ovunque si usi il path, mettere PATH_MAX oppure (meglio) far si che l'allocazione sia dinamica
    // TODO: stiamo usando ovunque path assoluti, dovremmo usare dei path relativi
    switch (argc)
    {
        case 1:
            inputLoop();
            break;
        default:
            E_Print(RED
                    "usage:"
                    RESET
                    " ./edix\n");
            exit(EXIT_FAILURE);
    }

    //TODO: messaggio di saluto

    return 0;
}
int inputLoop()
{
    size_t lineSize = 256;
    char *line = (char *) malloc(256 * sizeof(char));

    size_t bytesRead;
    Env env = HOMEPAGE;

    // while (((int) (bytesRead = getline(&line, &lineSize, stdin))) != -1)
    // {
    //     if (bytesRead == 1) continue;

    //     line[bytesRead - 1] = '\0';
    //     switch (env)
    //     {
    //         case HOMEPAGE:
    //             parseHome(line, &env);
    //             break;
    //         case PROJECT:
    //             parseProj(line, &env);
    //             break;
    //         case SETTINGS:
    //             parseSettings(line, &env);
    //             break;
    //         case EXIT:
    //             //Unreachable
    //             break;
    //     }
    //     if (env == EXIT)
    //         break;
    // }

    // free(line);
    while ((line = linenoise("==> ")) != nullptr)
    {
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
            break;
    }

    linenoiseFree(line);
    D_Print("Uscita in corso...\n");

    return 0;
}

