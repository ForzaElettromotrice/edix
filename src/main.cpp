#include "main.hpp"

int main(int argc, char *argv[])
{
    checkDb();
    banner();
    // TODO: magari puoi aprire direttamente un progetto passandolo come argomento
    switch (argc)
    {
        case 1:
            inputLoop();
            break;
        default:
            fprintf(stderr, RED "usage:" RESET " ./edix\n");
            exit(EXIT_FAILURE);
    }

    //TODO: fare tutte le cose da fare prima di chiudere


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

    return 0;
}

