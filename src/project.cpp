//
// Created by f3m on 07/01/24.
//

#include "project.hpp"

int parseProj(char *line, Env *env)
{
    /**
     *  16 comandi                                          <p>
     *  - exec          (Esegue frocerie)                   <p>
     *  - ls            (lista dei file)                    <p>
     *  - tree          (albero del progetto)               <p>
     *  - cd            (cambia ambiente di lavoro)         <p>
     *  - load          (carica un immagine)                <p>
     *  - rm            (rimuove un immagine)               <p>
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
    else if (strcmp(token, "exec") == 0)
        parseExec();
    else if (strcmp(token, "tree") == 0)
        parseTree();
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
    else
        printf(RED "Command not found\n" RESET);


    free(copy);
    return 0;
}

int parseExec()
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " exec frocName\n" RESET);
    }

    exec(path);

    return 0;
}
int parseLs()
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " ls [path ...]\n" RESET);
    }

    char *path_p = getStrFromKey((char *) "pPath");
    if (path_p == nullptr)
    {
        handle_error("");
    }

    int res = isPathIn(path, path_p);
    if (res != 0)
    {
        free(path_p);
        handle_error("Il path non si trova all'interno del progetto\n");
    }

    free(path_p);
    ls(path);

    return 0;
}
int parseTree()
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " tree [path ...]\n" RESET);
    }

    char *path_p = getStrFromKey((char *) "pPath");
    if (path_p == nullptr)
    {
        handle_error("");
    }
    int res = isPathIn(path, path_p);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto\n");
    }

    tree(path);

    return 0;
}
int parseCd()
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " cd nameDir\n" RESET);
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto\n");
    }

    cd(path);

    return 0;
}
int parseLoad()
{
    char *path = strtok(nullptr, " ");

    if (path != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " load pathToFile\n" RESET);
    }

    // Controlla che l'immagine sia valida
    if (isValidImage(path) == -1)
    {
        handle_error("I formati ammessi sono png/jpeg/ppm\n");
    }
    loadI(path);

    return 0;
}
int parseRm()
{
    char *path = strtok(nullptr, " ");

    if (path != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " rm filename\n" RESET);
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto\n");
    }

    rm(path);

    return 0;
}
int parseMkdir()
{
    char *name = strtok(nullptr, " ");

    if (name != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage:" BOLD ITALIC " mkdir nameDir\n" RESET);
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(name, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto\n");
    }

    mkdir(name);

    return 0;
}
int parseRmdir()
{
    char *name = strtok(nullptr, " ");

    if (name != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage:" BOLD ITALIC " rmdir nameDir\n" RESET);
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(name, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto\n");
    }

    rmdir(name);

    return 0;
}
int parseMv()
{
    char *pathSrc = strtok(nullptr, " ");
    char *pathDst = strtok(nullptr, " ");


    if ((pathSrc != nullptr && pathDst != nullptr) || strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " mv fromPath toPath\n" RESET);
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(pathDst, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto\n");
    }
    mv(pathSrc, pathDst);

    return 0;
}
int parseSett(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " settings\n" RESET);
    }

    settings(env);

    return 0;
}
int parseHelpP()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " help\n" RESET);
    }

    helpP();

    return 0;
}
int parseExitP(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " exit\n" RESET);
    }

    exitP(env);

    return 0;
}
int parseDix()
{
    char *op = strtok(nullptr, " ");
    char *name = strtok(nullptr, " ");

    if (op == nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD " dix operation [name]\n" RESET);
    }

    if (strcmp(op, "list") == 0 && name == nullptr)
        dixList();
    else if (strcmp(op, "commit") == 0 && name != nullptr)
        dixCommit(name);
    else if (strcmp(op, "reload") == 0 && name != nullptr)
        dixReload(name);
    else
    {
        handle_error("usage" BOLD " dix operation [name]\n" RESET);
    }

    return 0;
}
int parseForce()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " force\n" RESET);
    }

    force();
    return 0;
}


int exec(char *path)
{
    // TODO:
    return 0;
}
int ls(const char *path)
{

    char comm[256];

    sprintf(comm, "ls %s", path == nullptr ? "." : path);

    int status = system(comm);

    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando ls\n");
    }

    return 0;
}
int tree(char *path)
{
    char comm[256];

    sprintf(comm, "tree %s", path == nullptr ? "." : path);

    int status = system(comm);

    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando tree\n");
    }
    return 0;
}
int cd(char *path)
{

    char comm[256];

    // TODO: se non viene specificato il path, torni alla $HOME del progetto; va presa da redis
    sprintf(comm, "cd %s", path);

    int status = system(comm);

    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando cd\n");
    }

    return 0;
}
int loadI(char *path)
{
    char comm[256];

    if (path == nullptr)
    {
        handle_error("Il path non puo' essere null\n");
    }
    // TODO: Prendi da redis il percorso del progetto, sui cui si dovra' caricare l'immagine
    sprintf(comm, "cp %s %s", path, nullptr);

    int status = system(comm);

    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando load\n");
    }

    return 0;
}
int rm(char *name)
{
    char comm[256];

    if (name == nullptr)
    {
        handle_error("Il nome del file non puo' essere nullo\n");
    }

    sprintf(comm, "rm %s", name);

    int status = system(comm);

    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando rm\n");
    }
    return 0;

}
int mkdir(char *name)
{

    char comm[256];

    if (name == nullptr)
    {
        handle_error("Il nome della directory non puo' essere nullo\n");
    }

    sprintf(comm, "mkdir %s", name);

    int status = system(comm);

    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando mkdir\n");
    }
    return 0;
}
int rmdir(char *name)
{
    sprintf(name, "-r %s", name);

    rm(name);

    return 0;
}
int mv(char *fromPath, char *toPath)
{
    char comm[256];

    if (fromPath == nullptr || toPath == nullptr)
    {
        handle_error("Il nome del path non puo' essere nullo\n");
    }

    sprintf(comm, "mv %s %s", fromPath, toPath);

    int status = system(comm);

    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando mv\n");
    }

    return 0;
}
int settings(Env *env)
{
    //TODO:
    D_PRINT("settings\n");
    return 0;
}
int helpP()
{
    printf("Ecco la lista dei comandi che puoi utilizzare all'interno del tuo progetto:\n\n"
           YELLOW BOLD "  ls"    RESET " [path ...]\t\t\tStampa il contenuto della directory path. Se non viene inserito path, stampa il contenuto della directory corrente\n"
           YELLOW BOLD "  tree"  RESET " [path ...]\t\tStampa il contenuto della directory in un formato ad albero della directory path. Se non viene inserito path, stampa il contenuto della directory corrente\n"
           YELLOW BOLD "  exec"  RESET " nameFroc\t\t\tEsegui la froceria nameFroc\n"
           YELLOW BOLD "  cd"    RESET " nameDir\t\t\tCambia la directory corrente a nameDir\n"
           YELLOW BOLD "  loadI" RESET " pathToFile\t\tCarica l'immagine pathToFile\n"
           YELLOW BOLD "  rm"    RESET " filename ...\t\tRimuovi file filename\n"
           YELLOW BOLD "  mkdir" RESET " nameDir ...\t\tCrea la directory nameDir\n"
           YELLOW BOLD "  rmdir" RESET " nameDir ...\t\tRimuovi la directory nameDir\n"
           YELLOW BOLD "  mv"    RESET " source target\t\tRinomina il file source in target \n"
           YELLOW BOLD "  mv"    RESET " source ... nameDir\t\tSposta il file source alla directory nameDir\n"
           YELLOW BOLD "  sett"  RESET "\t\t\t\tAccedi ai settings\n"
           YELLOW BOLD "  exit"  RESET "\t\t\t\tEsci dal progetto\n");
    return 0;
}
int exitP(Env *env)
{
    //TODO: rimuovere tutta la roba da redis
    //TODO: cambia ch
    D_PRINT("Uscita dal progetto in corso...\n");
    *env = HOMEPAGE;
    return 0;
}
int dixList()
{
    char *projectName = getStrFromKey((char *) "Project");
    char *dixs = getDixs(projectName);
    if (dixs == nullptr)
    {
        free(projectName);
        free(dixs);
        return 1;
    }
    printf("%s\n", dixs);

    free(dixs);
    return 0;
}
int dixCommit(char *name)
{
    char *projectPath = getStrFromKey((char *) "pPath");
    if (checkDix(projectPath, name))
        return 1;

    char *comment = askComment();
    setElementToRedis((char *) "dixComments", comment);

    if (cloneProject(projectPath, projectPath, name))
    {
        handle_error("Errore nella clonazione del progetto!\n");
    }


    return 0;
}
int dixReload(char *name)
{
    // TODO:
    char *projectName = getStrFromKey((char *) "Project");
    if (loadDix(name, projectName))
    {
        free(projectName);
        return 1;
    }


    free(projectName);

    return 0;
}
int force()
{
    //TODO: testare
    D_PRINT("Forcing push...\n");
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
    return 0;
}


int isPathIn(const char *path, const char *pathProj)
{

    char *absPath = realpath(path, nullptr);
    if (absPath == nullptr)
    { handle_error("Errore nella risoluzione del percorso\n"); }

    int res = strncmp(absPath, pathProj, strlen(pathProj));

    return res;
}
int isValidImage(char *path)
{
    char *ext = strrchr(path, '.');

    if (ext == nullptr)
    {
        handle_error("Errore nella risoluzione del percorso\n");
    }

    if (strcmp(ext, ".png") != 0 || strcmp(ext, ".jpeg") != 0 || strcmp(ext, ".ppm") != 0)
        return -1;
    return 0;
}
char *askComment()
{

    size_t aSize = 500;
    char *line = (char *) malloc(256 * sizeof(char));

    printf(BOLD "Vuoi inserire un commento? [y/N] -> " RESET);
    uint bRead = getline(&line, &aSize, stdin);
    if (bRead == -1)
    {
        fprintf(stderr, RED "\nError: " RESET "Errore inserimento dati\n");
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

    D_PRINT("Il commento risultante finale è : \n\"\n%s\"\n", comment);

    return comment;

}
int checkDix(char *projectPath, char *dixName)
{
    char path[256];
    sprintf(path, "%s/.dix/%s", projectPath, dixName);

    DIR *dir = opendir(path);
    if (!dir)
    {
        sprintf(path, "mkdir -p %s", path);
        if (system(path) != 0)
            return 1;
    } else
    {
        handle_error("Dix già esistente!\n");
    }


    return 0;
}
int cloneProject(char *projectPath, char *path, char *dixName)
{
    DIR *dir = opendir(path);

    if (dir == nullptr)
    {
        handle_error("Errore nell'apertura della directory: %s\n", strerror(errno));
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        if(entry->d_name[0] == '.')
                continue;

        struct stat file_stat{};

        if (stat(path, &file_stat) == -1)
        {
            fprintf(stderr, RED "Error:" RESET "Errore nell'ottenere le informazioni sul file\n");
            continue;
        }
        if (S_ISDIR(file_stat.st_mode))
        {

            char *newPath = (char *)malloc(256 * sizeof(char));
            if (newPath == nullptr)
                return 1;
            sprintf(newPath, "%s/%s", path, entry->d_name);
            cloneProject(projectPath, newPath, dixName);
            free(newPath);
        } else
        {
            char *key = (char *)malloc(256 * sizeof(char));
            if (key == nullptr)
                return 1;

            sprintf(key, "%sImages", dixName);
            setElementToRedis(key, entry->d_name);
            sprintf(key, "%sPaths", dixName);
            setElementToRedis(key, path);

            free(key);

            char *command = (char *)malloc(256 * sizeof(char));
            if (command == nullptr)
                return 1;

            sprintf(command, "cp %s/%s %s/.dix/%s", path, entry->d_name, projectPath, dixName);
            system(command);
            free(command);
        }


    }
    free(entry);
    closedir(dir);

    return 0;
}
