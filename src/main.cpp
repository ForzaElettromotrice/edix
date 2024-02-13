#include "main.hpp"


int main(int argc, char *argv[])
{
    if (checkPostgresService() || checkRedisService())
        exit(EXIT_FAILURE);
    checkDb();
    deallocateFromRedis();
    banner();
    // TODO: ovunque si usi il path, mettere PATH_MAX oppure (meglio) far si che l'allocazione sia dinamica
    // TODO: stiamo usando ovunque path assoluti, dovremmo usare dei path relativ
    Env env;
    switch (argc)
    {
        case 1:
        {
            env = HOMEPAGE;
            inputLoop(env);
            break;
        }
        case 2:
        {
            env = HOMEPAGE;
            char tmp[50];
            sprintf(tmp, "open %s", argv[1]);
            parseHome(tmp, &env);
            inputLoop(env);
            break;
        }
        default:
            E_Print(RED "usage:" RESET " ./edix [projectName]\n");
            exit(EXIT_FAILURE);
    }
    printf(BOLD "Alla prossima %s ;)\n" RESET, getlogin());

    return 0;
}
int inputLoop(Env env)
{
    size_t lineSize = 256;
    char *line = (char *) malloc(256 * sizeof(char));

    size_t bytesRead;

    while (1)
    {
        print_prompt(env);
        if (((int) (bytesRead = getline(&line, &lineSize, stdin))) == -1)
            break;

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
            break;
    }

    free(line);
    //TODO: vediamo se riusciamo a far funzionare linenoise
    // while ((line = linenoise("==> ")) != nullptr)
    // {
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

    // linenoiseFree(line);
    D_Print("Uscita in corso...\n");

    return 0;
}

int print_prompt(Env env)
{
    char *cwd = (char *) malloc(sizeof(char) * 256);
    char *username = getlogin();

    if (getcwd(cwd, 256) == NULL)
    {
        E_Print("Errore con getcwd()\n");
        free(cwd);
        return 1;
    }
    char *cpd = strrchr(cwd, '/');
    if (cpd == NULL)
    {
        E_Print("Errore con strrchr\n");
        free(cwd);
        return 1;
    }

    switch (env)
    {
        case PROJECT:
        {
            printf(BOLD BLUE "%s" RESET BOLD "@" RESET GREEN BOLD "%s" RESET  BOLD "> " RESET, username, cpd + 1);
            break;
        }
        case HOMEPAGE:
        {
            printf(BOLD BLUE "%s" RESET BOLD "@" RESET YELLOW BOLD "%s" RESET BOLD "> " RESET, username, "edix-homepage");
            break;
        }
        case SETTINGS:
        {
            printf(BOLD BLUE "%s" RESET BOLD "@" RESET RED BOLD "%s-settings" RESET BOLD "> " RESET, username, cpd + 1, "settings");
            break;
        }
        default:
            break;
    }

    free(cwd);
    return 0;
}

