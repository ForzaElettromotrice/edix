//
// Created by f3m on 30/12/23.
//

#include "homepage.hpp"


int banner()
{
    printf(BOLD GREEN
           "    _/_/_/_/        _/  _/  _/      _/\n"
           "   _/          _/_/_/        _/  _/\n"
           "  _/_/_/    _/    _/  _/      _/\n"
           " _/        _/    _/  _/    _/  _/\n"
           "_/_/_/_/    _/_/_/  _/  _/      _/\n\n"
           RESET
           BOLD BLUE "Benvenuto su EdiX :).\n" RESET
           BOLD YELLOW "help\t\t\t" RESET
           ITALIC "Se hai bisogno di maggiori informazioni sui comandi\n" RESET
           BOLD YELLOW "exit" RESET " /" BOLD YELLOW " Ctrl + D\t\t" RESET
           ITALIC "Per uscire\n\n" RESET);
    return 0;
}
int checkDefaultFolder()
{
    char path[256];
    sprintf(path, "%s/EdixProjects", getenv("HOME"));
    DIR *defaultDir = opendir(path);

    if (!defaultDir)
    {
        D_Print("Creating default dir\n");
        system("mkdir ~/EdixProjects > /dev/null");
    }
    closedir(defaultDir);
    return 0;
}
int askParams(char *name, char *path, char *comp, char *tpp, char *tup, char *modEx, uint *tts, bool *backup)
{
    size_t aSize = 256;
    char *answer = (char *) malloc(256 * sizeof(char));
    size_t bRead;

    do
    {
        printf(BOLD YELLOW "Dove vuoi salvare il progetto? " RESET " (inserisci un path relativo o assoluto) (default: ~/EdixProjects): ");
        if ((bRead = getline(&answer, &aSize, stdin)) == -1)
        {
            free(answer);
            printf("\n");
            D_Print("Operazione annullata\n");
            return 1;
        }
        if (bRead == 1)
            sprintf(answer, "~/EdixProjects");
        else
            answer[bRead - 1] = '\0';
    } while (!isValidPath(answer));
    sprintf(path, "%s/%s", answer, name);

    do
    {
        printf(BOLD YELLOW "Che estensione vuoi abbiano le immagini che crei?\n" RESET "\t- PPM (Default)\n\t- PNG\n\t- JPEG\n-> ");
        if ((bRead = getline(&answer, &aSize, stdin)) == -1)
        {
            free(answer);
            printf("\n");
            D_Print("Operazione annullata\n");
            return 1;
        }
        if (bRead == 1)
            sprintf(answer, "PPM");
        else
            answer[bRead - 1] = '\0';
    } while (!isValidComp(answer));
    sprintf(comp, "%s", answer);

    do
    {
        printf(BOLD YELLOW "Che tipo di parallelismo vuoi usare?\n" RESET "\t- Serial (Default)\n\t- OMP\n\t- CUDA\n-> ");
        if ((bRead = getline(&answer, &aSize, stdin)) == -1)
        {
            free(answer);
            printf("\n");
            D_Print("Operazione annullata\n");
            return 1;
        }
        if (bRead == 1)
            sprintf(answer, "Serial");
        else
            answer[bRead - 1] = '\0';
    } while (!isValidTPP(answer));
    sprintf(tpp, "%s", answer);

    do
    {
        printf(BOLD YELLOW "Che algoritmo di upscaling vuoi usare?\n " RESET "\t- Bilinear (Default)\n\t- Bicubic\n-> ");
        if ((bRead = getline(&answer, &aSize, stdin)) == -1)
        {
            free(answer);
            printf("\n");
            D_Print("Operazione annullata\n");
            return 1;
        }
        if (bRead == 1)
            sprintf(answer, "Bilinear");
        else
            answer[bRead - 1] = '\0';

    } while (!isValidTUP(answer));
    sprintf(tup, "%s", answer);

    do
    {
        printf(BOLD YELLOW "Che modalità di esecuzione vuoi usare?\n" RESET "\t- Immediate (Default)\n\t- Programmed\n-> ");
        if ((bRead = getline(&answer, &aSize, stdin)) == -1)
        {
            free(answer);
            printf("\n");
            D_Print("Operazione annullata\n");
            return 1;
        }
        if (bRead == 1)
            sprintf(answer, "Immediate");
        else
            answer[bRead - 1] = '\0';

    } while (!isValidMode(answer));
    sprintf(modEx, "%s", answer);

    do
    {
        printf(BOLD YELLOW "Vuoi attivare il version control? " RESET "[y,N]: ");
        if ((bRead = getline(&answer, &aSize, stdin)) == -1)
        {
            free(answer);
            printf("\n");
            D_Print("Operazione annullata\n");
            return 1;
        }
        if (bRead == 1)
            sprintf(answer, "N");
        else
            answer[bRead - 1] = '\0';
    } while (!isValidBackup(answer));
    *backup = answer[0] == 'y' || answer[0] == 'Y';


    if (*backup)
    {
        do
        {
            printf(BOLD YELLOW "Ogni quante istruzioni vuoi salvare un backup del progetto? " RESET "(istruzioni >= 5) (default: 5): ");
            if ((bRead = getline(&answer, &aSize, stdin)) == -1)
            {
                free(answer);
                printf("\n");
                D_Print("Operazione annullata\n");
                return 1;
            }
            if (bRead == 1)
                sprintf(answer, "5");
            else
                answer[bRead - 1] = '\0';
        } while (!isValidTTS(answer));
        *tts = strtoul(answer, nullptr, 10);
    } else
        *tts = 5UL;


    free(answer);

    return 0;
}
bool isValidName(char *word)
{
    for (size_t i = 0; i < strlen(word) - 1; ++i)
        if (!isalnum(word[i]) && word[i] != '_')
        {
            fprintf(stderr, RED "Error: " RESET "Invalid character " BOLD ITALIC "%c\n" RESET, word[i]);
            return false;
        }

    return true;
}
bool isValidFlag(const char *flag)
{
    return flag[0] == '-' && flag[1] == 'm' && flag[2] == '\0';
}
bool isValidPath(char *path)
{
    D_Print("Checking path...\n");

    if (path[0] == '~')
    {
        char *home = getenv("HOME");
        char *tmp = (char *) malloc((strlen(path) + strlen(home) + 1) * sizeof(char));
        if (tmp == nullptr)
        {
            fprintf(stderr, RED "Error: " RESET "Error while malloc!\n");
            return false;
        }
        sprintf(tmp, "%s%s", home, path + 1);
        memmove(path, tmp, strlen(tmp) + 1);
    }

    char *result = realpath(path, nullptr);

    if (result == nullptr)
    {
        fprintf(stderr, RED "Error: " RESET "Path non valido!\n");
        return false;
    }

    char *tmp = (char *) realloc(path, (strlen(result) + 1) * sizeof(char *));
    if (tmp == nullptr)
    {
        free(result);
        fprintf(stderr, RED "Error: " RESET "Error while realloc!\n");
        return false;
    }
    path = tmp;
    sprintf(path, "%s", result);
    free(result);

    return true;
}
bool isValidComp(char *comp)
{
    if (strcmp(comp, "PPM") != 0 && strcmp(comp, "PNG") != 0 && strcmp(comp, "JPEG") != 0)
    {
        fprintf(stderr, RED "Error: " RESET "Risposta non valida!\n");
        return false;
    }
    return true;
}
bool isValidTPP(char *tpp)
{
    if (strcmp(tpp, "Serial") != 0 && strcmp(tpp, "OMP") != 0 && strcmp(tpp, "CUDA") != 0)
    {
        fprintf(stderr, RED "Error: " RESET "Risposta non valida!\n");
        return false;
    }
    return true;
}
bool isValidTUP(char *tup)
{
    if (strcmp(tup, "Bilinear") != 0 && strcmp(tup, "Bicubic") != 0)
    {
        fprintf(stderr, RED "Error: " RESET "Risposta non valida!\n");
        return false;
    }
    return true;
}
bool isValidMode(char *mode)
{
    if (strcmp(mode, "Immediate") != 0 && strcmp(mode, "Programmed") != 0)
    {
        fprintf(stderr, RED "Error: " RESET "Risposta non valida!\n");
        return false;
    }
    return true;
}
bool isValidTTS(char *tts)
{
    if (strtoul(tts, nullptr, 10) < 5)
    {
        fprintf(stderr, RED "Error: " RESET "Risposta non valida!\n");
        return false;
    }
    return true;
}
bool isValidBackup(char *backup)
{
    if (strcmp(backup, "y") != 0 && strcmp(backup, "n") != 0 && strcmp(backup, "Y") != 0 && strcmp(backup, "N") != 0)
    {
        fprintf(stderr, RED "Error: " RESET "Risposta non valida!\n");
        return false;
    }
    return true;
}


int parseHome(char *line, Env *env)
{
    /**
     *  6 comandi <p>
     *  - new Name -m (crea il progetto) (-m sta per manuale) <p>
     *  - open Name (apre il progetto) <p>
     *  - del Name  (elimina il progetto) <p>
     *  - list      (visualizza tutti i progetti) <p>
     *  - help      (lista dei comandi disponibili) <p>
     *  - exit      (esce da edix) <p>
     */

    char *copy = strdup(line);
    char *token = strtok(copy, " ");


    if (strcmp(token, "new") == 0)
        parseNew(env);
    else if (strcmp(token, "open") == 0)
        parseOpen(env);
    else if (strcmp(token, "del") == 0)
        parseDel();
    else if (strcmp(token, "list") == 0)
        parseListH();
    else if (strcmp(token, "help") == 0)
        parseHelpH();
    else if (strcmp(token, "exit") == 0)
        parseExitH(env);
    else
        fprintf(stderr, RED "Error: " RESET "Command not found\n");


    free(copy);
    return 0;
}
int parseNew(Env *env)
{

    char *token1 = strtok(nullptr, " ");
    char *token2 = strtok(nullptr, " ");

    if (token1 == nullptr || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC "new ProjectName [-m]\n" RESET);
        return 1;
    }

    if (token2 != nullptr)
    {
        char *name;

        if (isValidFlag(token1))
        {
            if (isValidName(token2))
                name = token2;
            else
            {
                E_Print("usage " BOLD ITALIC "new ProjectName [-m]\n" RESET);
                return 1;
            }
        } else if (isValidFlag(token2) && isValidName(token1))
        {
            name = token1;
        } else
        {
            E_Print("usage " BOLD ITALIC "new ProjectName [-m]\n" RESET);
            return 1;
        }

        newP(name, true, env);
        return 0;

    } else if (isValidName(token1))
    {
        newP(token1, false, env);
        return 0;
    }

    E_Print("usage " BOLD ITALIC "new ProjectName [-m]\n" RESET);
    return 1;
}
int parseOpen(Env *env)
{
    char *name = strtok(nullptr, " ");

    if (name == nullptr || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage " BOLD ITALIC "open ProjectName\n" RESET);
        return 1;
    }
    openP(name, env);

    return 0;
}
int parseDel()
{
    char *name = strtok(nullptr, " ");
    if (name == nullptr || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " del ProjectName\n" RESET);
        return 1;
    }
    delP(name);

    return 0;
}
int parseListH()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " listH\n" RESET);
        return 1;
    }

    listH();
    return 0;
}
int parseHelpH()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " helpH\n" RESET);
        return 1;
    }

    helpH();
    return 0;
}
int parseExitH(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " exitH\n" RESET);
        return 1;
    }

    exitH(env);
    return 0;
}


int newP(char *name, bool ask, Env *env)
{
    checkDefaultFolder();

    char path[256];
    char comp[5] = "PPM";
    char tpp[16] = "Serial";
    char tup[16] = "Bilinear";
    char modEx[16] = "Immediate";
    bool backup = false;
    uint tts = 5;
    sprintf(path, "%s/EdixProjects/%s", getenv("HOME"), name);

    if (ask && askParams(name, path, comp, tpp, tup, modEx, &tts, &backup))
        return 1;

    if (addProject(name, path, comp, tpp, tup, modEx, tts, backup))
        return EXIT_FAILURE;

    char command[256];
    sprintf(command, "mkdir %s > /dev/null 2> /dev/null", path);
    if (system(command) != 0)
        fprintf(stderr, RED
                        "Error: "
                        RESET
                        "Errore nella creazione della cartella!\n");

    if (chdir(path) != 0)
    {
        perror(RED
               "CHDIR Error"
               RESET);
        printf(YELLOW
               "Progetto creato, usare open per tentare di aprirlo\n"
               RESET);
        return EXIT_FAILURE;
    }

    if (loadProjectOnRedis(name))
    {
        E_Print(
                "Failed to load project on redis\n"
                YELLOW
                "Progetto creato, usare open per tentare di aprirlo\n"
                RESET);
        return 1;
    }

    *env = PROJECT;
    return 0;
}
int openP(char *name, Env *env)
{
    if (!existProject(name))
    {
        E_Print("Questo progetto non esiste!\n");
        return 1;
    }

    if (loadProjectOnRedis(name))
        return EXIT_FAILURE;

    char *path = getStrFromKey((char *) "pPath");
    if (path == nullptr)
        return 1;

    if (chdir(path) != 0)
    {
        E_Print("chdir: %s", strerror(errno));
        return 1;
    }

    D_Print("Entering %s...\n", name);
    *env = PROJECT;
    return 0;
}
int delP(char *name)
{

    if (!existProject(name))
    {
        E_Print("Questo progetto non esiste!\n");
        return 1;
    }

    if (delProject(name))
        return 1;


    return 0;
}
int listH()
{
    char *names = getProjects();
    printf("%s\n", names);

    free(names);
    return 0;
}
int helpH()
{
    printf("Ecco la lista dei comandi da poter eseguire qui sulla homepage:\n\n"
           BOLD YELLOW "  new\t" RESET "projectName\t\tCrea un nuovo progetto projectName\n"
           BOLD YELLOW "  open\t" RESET "projectName\t\tApri il progetto projectName\n"
           BOLD YELLOW "  del\t" RESET "projectName\t\tCancella il progetto projectName\n"
           BOLD YELLOW "  list" RESET "\t\t\t\tVisualizza tutti i progetti\n"
           BOLD YELLOW "  help" RESET "\t\t\t\tPer maggiori informazioni\n"
           BOLD YELLOW "  exit" RESET "\t\t\t\tEsci\n");
    return 0;
}
int exitH(Env *env)
{
    *env = EXIT;
    return 0;
}
