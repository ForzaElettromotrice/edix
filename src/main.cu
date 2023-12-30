#include "main.h"


int main(int argc, char *argv[])
{
    // TODO: setup di tutte le risorse necessarie (tipo redis, o comunicatore col db)
    // TODO: magari puoi aprire direttamente un progetto passandolo come argomento
    switch (argc)
    {
        case 1:
            inputLoop();
            break;
        default:
            fprintf(stderr, "usage: ./edix\n");
            exit(EXIT_FAILURE);
    }


    return 0;
}
int inputLoop()
{
    size_t lineSize;
    char *line;

    int readed;
    Env env = HOMEPAGE;

    while ((readed = getline(&line, &lineSize, stdin)) != -1)
    {
        line[readed] = '\0';
        switch (env)
        {
            case HOMEPAGE:
                parseHome(line, &env);
                break;
            case PROJECT:
                break;
            case SETTINGS:
                break;
        }
    }

    return 0;
}


