//
// Created by f3m on 07/01/24.
//

#include "project.hpp"


int isPathIn(const char *path, const char *pathProj)
{

    char *absPath = (char *) malloc((strlen(path) + PATH_MAX) * sizeof(char));
    sprintf(absPath, "%s/%s", getcwd(nullptr, 0), path);

    int res = strncmp(absPath, pathProj, strlen(pathProj));
    free(absPath);

    return res;
}
bool isValidImage(char *path)
{
    char *ext = strrchr(path, '.');

    if (ext == nullptr)
    {
        E_Print("Errore nella risoluzione del percorso\n");
        return false;
    }


    if (strcmp(ext, ".png") != 0 && strcmp(ext, ".jpeg") != 0 && strcmp(ext, ".ppm") != 0)
        return false;
    return true;
}
char *askComment()
{

    size_t aSize = 500;
    char *line = (char *) malloc(256 * sizeof(char));

    printf(BOLD "Vuoi inserire un commento? [y/N] -> " RESET);
    uint bRead = getline(&line, &aSize, stdin);
    if (bRead == -1)
    {
        E_Print(RED "\nError: " RESET "Errore inserimento dati\n");
        return nullptr;
    } else if (bRead == 1 || strcmp(line, "n\n") == 0 || strcmp(line, "N\n") == 0)
    {
        sprintf(line, "");
        return line;
    }

    char *comment = (char *) malloc(500 * sizeof(char));
    bool stop = false;
    sprintf(comment, "");

    printf(BOLD ITALIC "Scrivi il tuo commento (Max 500 caratteri). Per uscire 2 volte invio o Ctrl-D\n" RESET);
    while (((bRead = getline(&line, &aSize, stdin)) != -1 && strlen(comment) <= 500))
    {
        if (bRead == 1)
        {
            if (stop)
            {
                comment[strlen(comment) - 1] = '\0';
                break;
            } else
                stop = true;
        } else
            stop = false;

        strcat(comment, line);
    }
    comment[499] = '\0';

    return comment;

}
int checkDix(char *projectPath, char *dixName)
{
    char path[256];
    sprintf(path, "%s/.dix/%s", projectPath, dixName);

    DIR *dir = opendir(path);
    if (!dir)
    {
        char command[256];
        sprintf(command, "mkdir -p %s", path);

        if (system(command) != 0)
            return 1;
    } else
    {
        E_Print("Dix già esistente!\n");
        return 1;
    }

    return 0;
}
int cloneProject(char *projectPath, char *path, char *dixName)
{
    DIR *dir = opendir(path);

    if (dir == nullptr)
    {
        E_Print("Errore nell'apertura della directory: %s\n", strerror(errno));
        return 1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        if (entry->d_name[0] == '.')
            continue;

        struct stat file_stat{};
        char *newPath = (char *) malloc(256 * sizeof(char));
        if (newPath == nullptr)
            return 1;

        sprintf(newPath, "%s/%s", path, entry->d_name);

        if (stat(newPath, &file_stat) == -1)
        {
            E_Print(RED "Error:" RESET "Errore nell'ottenere le informazioni sul file\n");
            continue;
        }
        if (S_ISDIR(file_stat.st_mode))
        {
            cloneProject(projectPath, newPath, dixName);
        } else
        {


            char *key = (char *) malloc(256 * sizeof(char));
            if (key == nullptr)
                continue;
            sprintf(key, "%sImages", dixName);
            setElementToRedis(key, entry->d_name);
            sprintf(key, "%sPaths", dixName);
            setElementToRedis(key, path);


            char *command = (char *) malloc(
                    (strlen(path) + strlen(entry->d_name) + strlen(projectPath) + strlen(dixName) + 12) * sizeof(char));
            if (command == nullptr)
            {
                sprintf(key, "%sImages", dixName);
                removeKeyFromRedis(key);
                sprintf(key, "%sPaths", dixName);
                removeKeyFromRedis(key);
                free(key);
                continue;
            }
            sprintf(command, "cp %s/%s %s/.dix/%s", path, entry->d_name, projectPath, dixName);
            if (system(command) != 0)
            {
                free(command);
                free(key);
                continue;
            }
            free(command);
            free(key);

        }
        free(newPath);
    }
    free(entry);
    closedir(dir);

    return 0;
}


int parseProj(char *line, Env *env)
{
    /**
     *  17 comandi                                          <p>
     *  - funx          (Esegue frocerie)                   <p>
     *  - ls            (lista dei file)                    <p>
     *  - tree          (albero del progetto)               <p>
     *  - cd            (cambia ambiente di lavoro)         <p>
     *  - load          (carica un immagine)                <p>
     *  - rm            (rimuove un immagine)               <p>
     *  - ristretto     (apre un immagine)                  <p>
     *  - mkdir         (crea una cartella)                 <p>
     *  - rmdir         (rimuove una cartella)              <p>
     *  - mv            (sposta un immagine o cartella)     <p>
     *  - settings      (apre i settings)                   <p>
     *  - dix commit    (esegue il commit del dix)          <p>
     *  - dix reload    (ricarica un dix passato)           <p>
     *  - dix list      (elenca tutti i dix disponibili)    <p>
     *  - force         (forza il push su db)               <p>
     *  - help          (lista dei comandi disponibili)     <p>
     *  - exit          (esce dal progetto)                 <p>
     */
    char *copy = strdup(line);
    char *token = strtok(copy, " ");


    if (strcmp(token, "ls") == 0)
        parseLs();
    else if (strcmp(token, "funx") == 0)
        parseFunx();
    else if (strcmp(token, "cd") == 0)
        parseCd();
    else if (strcmp(token, "load") == 0)
        parseLoad();
    else if (strcmp(token, "rm") == 0)
        parseRm();
    else if (strcmp(token, "mkdir") == 0)
        parseMkdir();
    else if (strcmp(token, "rmdir") == 0)
        parseRmdir();
    else if (strcmp(token, "mv") == 0)
        parseMv();
    else if (strcmp(token, "settings") == 0)
        parseSett(env);
    else if (strcmp(token, "help") == 0)
        parseHelpP();
    else if (strcmp(token, "exit") == 0)
        parseExitP(env);
    else if (strcmp(token, "dix") == 0)
        parseDix();
    else if (strcmp(token, "force") == 0)
        parseForce();
    else if (strcmp(token, "ristretto") == 0)
        parseRistretto();
    else
        printf(RED "Command not found\n" RESET);


    free(copy);
    return 0;
}

int parseRistretto()
{
    char *path = strtok(nullptr, " ");

    if (path == nullptr || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " ristretto path\n" RESET);
        return 1;
    }

    if (!isValidImage(path))
    {
        E_Print("I formati ammessi sono png/jpeg/ppm\n");
        return 1;
    }

    ristretto(path);

    return 0;
}

int parseFunx()
{
    char *name = strtok(nullptr, " ");
    if (name == nullptr)
    {
        E_Print("usage" BOLD ITALIC " funx FUNX ARGS ...\n" RESET);
        return 1;
    }

    char *args = name + strlen(name) + 1;
    if (args == nullptr)
    {
        E_Print("usage" BOLD ITALIC " funx FUNX ARGS ...\n" RESET);
        return 1;
    }

    funx(name, args);

    return 0;
}
int parseLs()
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " ls [path ...]\n" RESET);
        return 1;
    }

    if (path == nullptr)
        return ls(nullptr);

    char *path_p = getStrFromKey((char *) "pPath");
    if (path_p == nullptr)
    {
        E_Print("Path del progetto inesistente!\n");
        return 1;
    }

    int res = isPathIn(path, path_p);
    if (res != 0)
    {
        free(path_p);
        E_Print("Il path non si trova all'interno del progetto\n");
        return 1;
    }

    free(path_p);
    ls(path);

    return 0;
}
int parseCd()
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " cd nameDir\n" RESET);
        return 1;
    }

    if (path == nullptr)
        return cd(nullptr);

    char *pPath = getStrFromKey((char *) "pPath");
    if (pPath == nullptr)
        return 1;

    int res = isPathIn(path, pPath);
    if (res != 0)
    {
        E_Print("Il path non si trova all'interno del progetto\n");
        return 1;
    }

    cd(path);

    return 0;
}
int parseLoad()
{
    char *path = strtok(nullptr, " ");

    if (path == nullptr || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " load pathToFile\n" RESET);
        return 1;
    }

    if (!isValidImage(path))
    {
        E_Print("I formati ammessi sono png/jpeg/ppm\n");
        return 1;
    }
    loadI(path);

    return 0;
}
int parseRm()
{
    char *path = strtok(nullptr, " ");

    if (path == nullptr || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " rm filename\n" RESET);
        return 1;
    }


    char *pPath = getStrFromKey((char *) "pPath");
    if (pPath == nullptr)
        return 1;
    int res = isPathIn(path, pPath);
    if (res != 0)
    {
        E_Print("Il path non si trova all'interno del progetto\n");
        return 1;
    }

    rm(path);

    return 0;
}
int parseMkdir()
{
    char *name = strtok(nullptr, " ");
    char *err = strtok(nullptr, " ");

    if (name == nullptr || err != nullptr)
    {
        E_Print("usage:" BOLD ITALIC " mkdir nameDir\n" RESET);
        return 1;
    }

    char *pPath = getStrFromKey((char *) "pPath");
    if (pPath == nullptr)
        return 1;
    int res = isPathIn(name, pPath);
    if (res != 0)
    {
        E_Print("Il path non si trova all'interno del progetto\n");
        return 1;
    }

    mkdir(name);

    return 0;
}
int parseRmdir()
{
    char *name = strtok(nullptr, " ");

    if (name == nullptr || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage:" BOLD ITALIC " rmdir nameDir\n" RESET);
        return 1;
    }

    char *pPath = getStrFromKey((char *) "pPath");
    if (pPath == nullptr)
        return 1;
    int res = isPathIn(name, pPath);
    if (res != 0)
    {
        E_Print("Il path non si trova all'interno del progetto\n");
        return 1;
    }

    rmdir(name);

    return 0;
}
int parseMv()
{
    char *pathSrc = strtok(nullptr, " ");
    char *pathDst = strtok(nullptr, " ");


    if (pathSrc == nullptr || pathDst == nullptr || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " mv fromPath toPath\n" RESET);
        return 1;
    }


    char *pPath = getStrFromKey((char *) "pPath");
    int res = isPathIn(pathSrc, pPath);
    if (res != 0)
    {
        E_Print("Il path non si trova all'interno del progetto\n");
        return 1;
    }
    res = isPathIn(pathDst, pPath);
    if (res != 0)
    {
        E_Print("Il path non si trova all'interno del progetto\n");
        return 1;
    }

    mv(pathSrc, pathDst);
    free(pPath);

    return 0;
}
int parseSett(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " settings\n" RESET);
        return 1;
    }

    settings(env);

    return 0;
}
int parseDix()
{
    char *op = strtok(nullptr, " ");
    char *name = strtok(nullptr, " ");

    if (op == nullptr || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD " dix operation [name]\n" RESET);
        return 1;
    }

    if (strcmp(op, "list") == 0 && name == nullptr)
        dixList();
    else if (strcmp(op, "commit") == 0 && name != nullptr)
        dixCommit(name);
    else if (strcmp(op, "reload") == 0 && name != nullptr)
        dixReload(name);
    else
    {
        E_Print("usage" BOLD " dix operation [name]\n" RESET);
        return 1;
    }

    return 0;
}
int parseForce()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " force\n" RESET);
        return 1;
    }

    force();
    return 0;
}
int parseHelpP()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " help\n" RESET);
        return 1;
    }

    helpP();

    return 0;
}
int parseExitP(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " exit\n" RESET);
        return 1;
    }

    exitP(env);

    return 0;
}
int funx(char *name, char *args)
{
    if (strcmp(name, "testBlur") == 0)
        parseBlurArgs(args);
    else if (strcmp(name, "grayscale") == 0)
        parseGrayscaleArgs(args);
    else if (strcmp(name, "colorfilter") == 0)
        parseColorFilterArgs(args);
    else if (strcmp(name, "upscale") == 0)
        parseUpscaleArgs(args);
    else if (strcmp(name, "downscale") == 0)
        parseDownscaleArgs(args);
    else if (strcmp(name, "overlap") == 0)
        parseOverlapArgs(args);
    else if (strcmp(name, "composition") == 0)
        parseCompositionArgs(args);
    else
    {
        E_Print("Funx non trovata\n");
        return 1;
    }

    return 0;
}
int ls(const char *path)
{
    path = path == nullptr ? "." : path;
    char *comm = (char *) malloc((strlen(path) + 4) * sizeof(char));
    if (comm == nullptr)
    {
        E_Print("Error while malloc!\n");
        return 1;
    }

    sprintf(comm, "ls --color=auto %s", path);

    if (system(comm))
    {
        free(comm);
        E_Print("Errore nell'esecuzione del comando ls\n");
        return 1;
    }
    free(comm);
    return 0;
}
int cd(char *path)
{
    bool toFree = path == nullptr;
    if (toFree)
    {
        path = getStrFromKey((char *) "pPath");
    }


    if (chdir(path) != 0)
    {
        if (toFree)
            free(path);
        E_Print("Errore nell'esecuzione del comando cd\n");
        return 1;
    }

    if (toFree)
        free(path);
    return 0;
}
int loadI(char *path)
{
    char *pPath = getStrFromKey((char *) "pPath");
    if (pPath == nullptr)
        return 1;
    char *comm = (char *) malloc((strlen(path) + strlen(pPath) + 4));
    if (comm == nullptr)
    {
        free(pPath);
        E_Print("Error while malloc!\n");
        return 1;
    }
    sprintf(comm, "cp %s %s", path, pPath);

    if (system(comm))
    {
        free(comm);
        E_Print("Errore nell'esecuzione del comando load\n");
        return 1;
    }

    free(pPath);
    free(comm);
    return 0;
}
int rm(char *name)
{
    char *comm = (char *) malloc((strlen(name) + 4) * sizeof(char));
    if (comm == nullptr)
    {
        E_Print("Error while malloc!\n");
        return 1;
    }

    sprintf(comm, "rm -rf %s", name);

    if (system(comm))
    {
        free(comm);
        E_Print("Errore nell'esecuzione del comando rm\n");
        return 1;
    }
    free(comm);
    return 0;

}
int mkdir(char *name)
{

    char *comm = (char *) malloc((strlen(name) + 7) * sizeof(char));
    if (comm == nullptr)
    {
        E_Print("Error while malloc!");
        return 1;
    }

    sprintf(comm, "mkdir -p %s", name);

    if (system(comm))
    {
        free(comm);
        E_Print("Errore nell'esecuzione del comando mkdir\n");
        return 1;
    }
    free(comm);
    return 0;
}
int rmdir(char *name)
{
    char *path = (char *) malloc((strlen(name) + 4) * sizeof(char));
    sprintf(path, "-r %s", name);

    rm(name);

    free(path);

    return 0;
}
int mv(char *fromPath, char *toPath)
{
    char *comm = (char *) malloc((strlen(fromPath) + strlen(toPath) + 5) * sizeof(char));
    if (comm == nullptr)
    {
        E_Print("Error while malloc!\n");
        return 1;
    }

    sprintf(comm, "mv %s %s", fromPath, toPath);

    if (system(comm))
    {
        free(comm);
        E_Print("Errore nell'esecuzione del comando mv\n");
        return 1;
    }
    free(comm);

    return 0;
}
int settings(Env *env)
{
    *env = SETTINGS;
    D_Print("Entering settings...\n");
    return 0;
}
int dixList()
{
    force();
    char *projectName = getStrFromKey((char *) "Project");
    if (projectName == nullptr)
        return 1;
    char *dixs = getDixs(projectName);
    if (dixs == nullptr)
    {
        free(projectName);
        return 1;
    }
    printf("%s\n", dixs);

    free(projectName);
    free(dixs);
    return 0;
}
int dixCommit(char *name)
{
    char *projectPath = getStrFromKey((char *) "pPath");
    if (checkDix(projectPath, name))
        return 1;


    char *comment = askComment();
    setElementToRedis((char *) "dixNames", name);
    setElementToRedis((char *) "dixComments", comment);

    cloneProject(projectPath, projectPath, name);


    return 0;
}
int dixReload(char *name)
{
    force();

    char *pPath = getStrFromKey((char *) "pPath");
    if (pPath == nullptr)
        return 1;

    char *comm = (char *) malloc((strlen(pPath) + 8) * sizeof(char));
    if (comm == nullptr)
    {
        free(pPath);
        E_Print("Error while malloc!\n");
        return 1;
    }
    sprintf(comm, "rm -rf %s/*", pPath);
    if (system(comm))
    {
        free(comm);
        free(pPath);
        E_Print("Errore nell'esecuzione del comando rm\n");
        return 1;
    }
    free(comm);


    char *projectName = getStrFromKey((char *) "Project");
    if (projectName == nullptr)
    {
        free(pPath);
        return 1;
    }

    loadDix(name, projectName, pPath);

    return 0;
}
int force()
{
    D_Print("Forcing push...\n");
    char **names = getCharArrayFromRedis((char *) "dixNames");
    char **comments = getCharArrayFromRedis((char *) "dixComments");
    char *projectName = getStrFromKey((char *) "pName");


    for (int i = 0; names[i] != nullptr; ++i)
    {
        char key[256];
        sprintf(key, "%sPaths", names[i]);
        char **paths = getCharArrayFromRedis(key);
        sprintf(key, "%sImages", names[i]);
        char **images = getCharArrayFromRedis(key);

        addDix(projectName, names[i], comments[i], images, paths);

        for (int j = 0; paths[j] != nullptr; ++j)
        {
            free(paths[j]);
            free(images[j]);
        }
        free(paths);
        free(images);
        free(names[i]);
        free(comments[i]);
    }
    free(names);
    free(comments);
    free(projectName);

    delDixFromRedis();

    int id = getIntFromKey((char *) "ID");
    char *tup = getStrFromKey((char *) "TUP");
    char *mode = getStrFromKey((char *) "Mode");
    char *comp = getStrFromKey((char *) "COMP");
    uint tts = (uint) getIntFromKey((char *) "TTS");
    char *tpp = getStrFromKey((char *) "TPP");
    bool backup = getIntFromKey((char *) "Backup") == 1;
    char *pName = getStrFromKey((char *) "pName");
    updateSettings(id, tup, mode, comp, tts, tpp, backup, pName);

    return 0;
}
int helpP()
{
    printf("Ecco la lista dei comandi che puoi utilizzare all'interno del tuo progetto:\n\n"
           YELLOW BOLD "  ls"           RESET " [" UNDERLINE "OPTION" RESET "] ... [" UNDERLINE "FILE" RESET "] ...\t\t\t\tStampa il contenuto di FILE. Se non viene inserito FILE, stampa il contenuto della directory corrente\n"
           YELLOW BOLD "  funx"         RESET " " UNDERLINE "FUNX" RESET " " UNDERLINE "ARGS" RESET  " ...\t\t\t\t\tEsegui la funx FUNX con ARGS\n"
           YELLOW BOLD "  cd"           RESET " [" UNDERLINE "DIRECTORY" RESET "]\t\t\t\t\tCambia la directory corrente a DIRECTORY\n"
           YELLOW BOLD "  load"         RESET " " UNDERLINE "FILE" RESET " ...\t\t\t\t\t\tCarica l'immagine FILE\n"
           YELLOW BOLD "  rm"           RESET " [" UNDERLINE "OPTION" RESET "] ... [" UNDERLINE "FILE" RESET "] ...\t\t\t\tRimuovi FILE\n"
           YELLOW BOLD "  mkdir"        RESET " [" UNDERLINE "OPTION" RESET "] ... [" UNDERLINE "FILE" RESET "] ...\t\t\t\tCrea la directory DIRECTORY\n"
           YELLOW BOLD "  rmdir"        RESET " [" UNDERLINE "OPTION" RESET "] ... [" UNDERLINE "FILE" RESET "] ...\t\t\t\tRimuovi la directory DIRECTORY\n"
           YELLOW BOLD "  mv"           RESET " [" UNDERLINE "OPTION" RESET "] ... " UNDERLINE "SOURCE" RESET " " UNDERLINE "DEST" RESET "\t\t\t\tRinomina SOURCE in DEST \n"
           YELLOW BOLD "  mv"           RESET " [" UNDERLINE "OPTION" RESET "] ... " UNDERLINE "SOURCE" RESET " ... " UNDERLINE "DIRECTORY" RESET "\t\t\tSposta SOURCE in DIRECTORY\n"
           YELLOW BOLD "  ristretto"    RESET " " UNDERLINE "FILE" RESET "\t\t\t\t\t\tApre l'immagine FILE\n"
           YELLOW BOLD "  force"        RESET "\t\t\t\t\t\t\tForza il caricamento delle modifiche su DB\n"
           YELLOW BOLD "  settings"     RESET "\t\t\t\t\t\tAccedi ai settings\n"
           YELLOW BOLD "  dix commmit"  RESET "\t\t\t\t\t\tEsegue il commit del dix\n"
           YELLOW BOLD "  dix reload"   RESET "\t\t\t\t\t\tRicarica un dix commitato precedentemente\n"
           YELLOW BOLD "  dix list"     RESET "\t\t\t\t\t\tElenca tutti i dix commitati finora\n"
           YELLOW BOLD "  help"         RESET "\t\t\t\t\t\t\tElenca la lista dei comandi da poter eseguire\n"
           YELLOW BOLD "  helpF"        RESET "\t\t\t\t\t\t\tElenca la lista delle funx da poter eseguire\n"
           YELLOW BOLD "  exit"         RESET "\t\t\t\t\t\t\tEsci dal progetto\n"
           "----------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
           "Ecco la liste delle funx che puoi utilizzare sulle tue immagini:\n\n"
           BOLD GREEN "   funx testBlur " RESET UNDERLINE "IN" RESET " " UNDERLINE "OUT" RESET " " UNDERLINE "RADIUS" RESET ITALIC BOLD "\t\t\t\tEffettua il testBlur di IN con radius RADIUS e viene salvato in OUT\n\t" RESET "* " BOLD "IN" RESET ": il path dell'immagine di input\n\t* " BOLD "OUT" RESET ": il path dell'immagine in output\n\t* " BOLD "RADIUS" RESET ": il livello di testBlur (> 0)\n\n"
           BOLD GREEN "   funx colorfilter " RESET UNDERLINE "IN" RESET " " UNDERLINE "OUT" RESET " " UNDERLINE "R" RESET " " UNDERLINE "G" RESET " " UNDERLINE "B" RESET " " UNDERLINE "TOLERANCE" RESET ITALIC BOLD"\t\tEffettua il color filter di IN di valore RGB con tolerance TOLERANCE e viene salvato in OUT\n\t" RESET "* " BOLD "IN" RESET ": il path dell'immagine di input\n\t* " BOLD "OUT" RESET ": il path dell'immagine in output\n\t* " BOLD "R" RESET ": il valore di rosso (0 <= R <= 255)\n\t* " BOLD "G" RESET ": il valore di verde (0 <= G <= 255)\n\t* " BOLD "B" RESET ": il valore di blu (0 <= B <= 255)\n\t* " BOLD "TOLERANCE" RESET ": permette di regolare la sensibilità del filtro rispetto i dettagli del colore nell'immagine(>=0)\n\n"
           BOLD GREEN "   funx grayscale " RESET UNDERLINE "IN" RESET " " UNDERLINE "OUT" RESET ITALIC BOLD"\t\t\t\tEffettua il grayscale di IN e viene salvato in OUT \n\t" RESET "* " BOLD "IN" RESET ": il path dell'immagine di input\n\t* " BOLD "OUT" RESET ": il path dell'immagine in output\n\n"
           BOLD GREEN "   funx composition " RESET UNDERLINE "IN1" RESET " " UNDERLINE "IN2" RESET " " UNDERLINE "OUT" RESET " " UNDERLINE "SIDE" RESET ITALIC BOLD"\t\t\tCompone IN2 sul SIDE di IN1 e il risultato viene salvato in OUT\n\t" RESET "* " BOLD "IN1" RESET ": il path dell'immagine di input\n\t* " BOLD "IN2" RESET ": il path dell'immagine da comporre a IN1\n\t* " BOLD "OUT" RESET ": il path dell'immagine in output\n\t* " BOLD "SIDE" RESET ": indica il lato su cui comporre l'immagine (UP=0, DOWN=1, LEFT=2, RIGTH=3)\n\n"
           BOLD GREEN "   funx overlap " RESET UNDERLINE "IN1" RESET " " UNDERLINE "IN2" RESET " " UNDERLINE "OUT" RESET " " UNDERLINE "X" RESET " " UNDERLINE "Y" RESET ITALIC BOLD "\t\t\t\tViene sovrapposta IN2 su IN1 alla posizione (X, Y) e viene salvato in OUT \n\t" RESET "* " BOLD "IN1" RESET ": il path dell'immagine di input\n\t* " BOLD "IN2" RESET ": il path dell'immagine da sovrapporre a IN1\n\t* " BOLD "OUT" RESET ": il path dell'immagine in output\n\t* " BOLD "X" RESET ": posizione sull'ascisse\n\t* " BOLD "Y" RESET ": posizione sull'ordinata\n\n"
           BOLD GREEN "   funx upscale " RESET UNDERLINE "IN" RESET " " UNDERLINE "OUT" RESET " " UNDERLINE "FACTOR" RESET ITALIC BOLD "\t\t\t\tEffettua l'upscale di IN di fattore FACTOR e viene salvato in OUT\n\t" RESET " * " BOLD "IN" RESET ": il path dell'immagine di input\n\t * " BOLD "OUT" RESET ": il path dell'immagine in output\n\t * " BOLD "FACTOR" RESET ": fattore di scala dell'immagine(> 0)\n\n"
           BOLD GREEN "   funx downscale " RESET UNDERLINE "IN" RESET " " UNDERLINE "OUT" RESET " " UNDERLINE "FACTOR" RESET ITALIC BOLD "\t\t\t\tEffettua il downscale di IN di fattore FACTOR e viene salvato in OUT\n\t" RESET " * " BOLD "IN" RESET ": il path dell'immagine di input\n\t * " BOLD "OUT" RESET ": il path dell'immagine in output\n\t *" BOLD " FACTOR" RESET ": fattore di scala dell'immagine(> 0)\n\n");
    return 0;
}
int exitP(Env *env)
{
    force();
    deallocateFromRedis();

    D_Print("Uscita dal progetto in corso...\n");
    *env = HOMEPAGE;
    return 0;
}

int ristretto(char *path)
{
    char *comm = (char *) malloc((strlen(path) + 10) * sizeof(char));
    if (comm == nullptr)
    {
        E_Print("Error while malloc!\n");
        return 1;
    }

    sprintf(comm, "ristretto %s", path);

    if (system(comm))
    {
        free(comm);
        E_Print("Errore nell'esecuzione del comando ristretto\n");
        return 1;
    }
    free(comm);
    return 0;
}
