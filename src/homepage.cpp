//
// Created by f3m on 30/12/23.
//

#include "homepage.hpp"


bool isValidName(char *word)
{
    for (size_t i = 0; i < strlen(word) - 1; ++i)
    {
        if (!isalnum(word[i]) && word[i] != '_')
        {
            //TODO: fare i print carini
            D_PRINT("Invalid character %c\n", word[i]);
            return false;
        }
    }

    return true;
}
bool isValidFlag(const char *flag)
{
    return flag[0] == '-' && flag[1] == 'm';
}
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
           BOLD YELLOW "new" RESET " projectName\t\t"
           ITALIC "Se e' la tua prima volta crea un progetto tramite il comando\n" RESET
           BOLD YELLOW "help\t\t\t" RESET
           ITALIC "Se hai bisogno di maggiori informazioni sui comandi, esegui\n" RESET
           BOLD YELLOW "exit" RESET " /" BOLD YELLOW " Ctrl + D\t\t" RESET
           ITALIC "Per uscire\n" RESET);
    return 0;
}
int askParams(char *name, char *path, char *comp, char *tpp, char *tup, char *modEx, uint *tts, bool *vcs)
{
    size_t aSize = 256;
    char *answer = (char *) malloc(256 * sizeof(char));
    size_t bRead;

    printf("Dove vuoi salvare il progetto? (inserisci un path relativo o assoluto): ");
    if ((bRead = getline(&answer, &aSize, stdin)) == -1)
    {
        fprintf(stderr, "Bad answer!\n");
        return 1;
    }
    //TODO: controllare
    answer[bRead - 1] = '\0';
    sprintf(path, "%s/%s", answer, name);

    printf("Che estenzione vuoi abbiano le immagini che crei?\n\t- PPM\n\t- PNG\n\t- JPEG\n-> ");
    if ((bRead = getline(&answer, &aSize, stdin)) == -1)
    {
        fprintf(stderr, "Bad answer!\n");
        return 1;
    }
    //TODO: controllare
    answer[bRead - 1] = '\0';
    sprintf(comp, "%s", answer);

    printf("Che tipo di parallelismo vuoi usare?\n\t- Serial\n\t- OMP\n\t- CUDA\n-> ");
    if ((bRead = getline(&answer, &aSize, stdin)) == -1)
    {
        fprintf(stderr, "Bad answer!\n");
        return 1;
    }
    //TODO: controllare
    answer[bRead - 1] = '\0';
    sprintf(tpp, "%s", answer);

    printf("Che algoritmo di upscaling vuoi usare?\n\t- Bilinear\n\t- Bicubic\n-> ");
    if ((bRead = getline(&answer, &aSize, stdin)) == -1)
    {
        fprintf(stderr, "Bad answer!\n");
        return 1;
    }
    //TODO: controllare
    answer[bRead - 1] = '\0';
    sprintf(tup, "%s", answer);

    printf("Che modalità di esecuzione vuoi usare?\n\t- Immediate\n\t- Programmed\n-> ");
    if ((bRead = getline(&answer, &aSize, stdin)) == -1)
    {
        fprintf(stderr, "Bad answer!\n");
        return 1;
    }
    //TODO: controllare
    answer[bRead - 1] = '\0';
    sprintf(modEx, "%s", answer);

    printf("Ogni quanto tempo vuoi salvare un backup del progetto? (secondi): ");
    if ((bRead = getline(&answer, &aSize, stdin)) == -1)
    {
        fprintf(stderr, "Bad answer!\n");
        return 1;
    }
    //TODO: controllare
    answer[bRead - 1] = '\0';
    *tts = strtoul(answer, nullptr, 10);

    printf("Vuoi attivare il version control? [y,n]: ");
    if ((bRead = getline(&answer, &aSize, stdin)) == -1)
    {
        fprintf(stderr, "Bad answer!\n");
        return 1;
    }
    //TODO: controllare
    answer[bRead - 1] = '\0';
    *vcs = strcmp(answer, "y") == 0;

    return 0;
}
int checkDefaultFolder()
{
    char path[256];
    sprintf(path, "%s/EdixProjects", getenv("HOME"));
    DIR *defaultDir = opendir(path);

    if (!defaultDir)
    {
        D_PRINT("Creating default dir\n");
        system("mkdir ~/EdixProjects > /dev/null");
    }
    closedir(defaultDir);
    return 0;
}

int parseHome(char *line, Env *env)
{
    /**
     *  6 comandi <p>
     *  - new Name -m (crea il progetto) (-m sta per manuale) <p>
     *  - open Name (apre il progetto) <p>
     *  - del Name  (elimina il progetto) <p>
     *  - view      (visualizza tutti i progetti) <p>
     *  - helpH      (lista dei comandi disponibili) <p>
     *  - exitH      (esce da edix) <p>
     */

    char *copy = strdup(line);
    char *token = strtok(copy, " ");


    if (strcmp(token, "new") == 0)
        parseNew(env);
    else if (strcmp(token, "open") == 0)
        parseOpen(env);
    else if (strcmp(token, "del") == 0)
        parseDel();
    else if (strcmp(token, "view") == 0)
        parseView();
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

    char *err = strtok(nullptr, " ");


    if (token1 == nullptr || (token2 != nullptr && err != nullptr))
    {
        handle_error("usage" BOLD ITALIC "new ProjectName [-m]\n" RESET);
    }

    if (token2 != nullptr)
    {
        char *name;

        if (isValidName(token1) && isValidFlag(token2))
        {
            name = token1;
        } else if (isValidName(token2) && isValidFlag(token1))
        {
            name = token2;
        } else
        {
            handle_error("usage " BOLD ITALIC "new ProjectName [-m]\n" RESET);
        }

        newP(name, true, env);
        return 0;

    } else if (isValidName(token1))
    {
        newP(token1, false, env);
        return 0;
    }

    handle_error("usage " BOLD ITALIC "new ProjectName [-m]\n" RESET);
}
int parseOpen(Env *env)
{
    char *name = strtok(nullptr, " ");

    if (name == nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage " BOLD ITALIC "open ProjectName\n" RESET);
    }
    openP(name, env);

    return 0;
}
int parseDel()
{
    char *name = strtok(nullptr, " ");
    if (name == nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " del ProjectName\n" RESET);
    }
    delP(name);

    return 0;
}
int parseView()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " view\n" RESET);
    }

    view();
    return 0;
}
int parseHelpH()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " helpH\n" RESET);
    }

    helpH();
    return 0;
}
int parseExitH(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " exitH\n" RESET);
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
    uint tts = 600;
    bool vcs = false;
    sprintf(path, "%s/EdixProjects/%s", getenv("HOME"), name);

    if (ask && askParams(name, path, comp, tpp, tup, modEx, &tts, &vcs))
    {
        handle_error("Errore inserimento parametri!\n");
    }

    if (addProject(name, path, comp, tpp, tup, modEx, tts, vcs))
        return EXIT_FAILURE;

    char command[256];
    sprintf(command, "mkdir %s > /dev/null", path);
    system(command);

    if (chdir(path) != 0)
    {
        perror("Errore durante il cambio della working directory");
        printf(YELLOW "Progetto creato, usare open per tentare di aprirlo\n" RESET);
        return EXIT_FAILURE;
    }

    if (loadProjectOnRedis(name))
    {
        handle_error(
                "Failed to load project on redis\n" YELLOW "Progetto creato, usare open per tentare di aprirlo\n" RESET);
    }

    *env = PROJECT;
    return 0;
}
int openP(char *name, Env *env)
{
    if (!existProject(name))
    {
        handle_error("Questo progetto non esiste!\n");
    }

    if (loadProjectOnRedis(name))
        return EXIT_FAILURE;

    D_PRINT("Entering %s...\n", name);
    *env = PROJECT;
    return 0;
}
int delP(char *name)
{

    if (!existProject(name))
    {
        handle_error("Questo progetto non esiste!\n");
    }

    if (delProject(name))
        return 1;


    return 0;
}
int view()
{
    char *names = getProjects();
    printf("%s\n", names);

    return 0;
}
int helpH()
{
    printf("Ecco la lista dei comandi da poter eseguire qui sulla homepage:\n\n"
           BOLD YELLOW "  new\t" RESET "nameProject\t\tCrea un nuovo progetto nameProject\n"
           BOLD YELLOW "  open\t" RESET "nameProject\t\tApri il progetto nameProject\n"
           BOLD YELLOW "  del\t" RESET "nameProject\t\tCancella il progetto nameProject\n"
           BOLD YELLOW "  help" RESET "\t\t\t\tPer maggiori informazioni\n"
           BOLD YELLOW "  view" RESET "\t\t\t\tVisualizza tutti i progetti\n"
           BOLD YELLOW "  exit" RESET "\t\t\t\tEsci\n");
    return 0;
}
int exitH(Env *env)
{
    D_PRINT("Uscita in corso...\n");
    *env = EXIT;
    return 0;
}
