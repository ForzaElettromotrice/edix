//
// Created by f3m on 07/01/24.
//

#include "project.hpp"

int parseProj(char *line, Env *env)
{
    /**
     *  12 comandi                                      <p>
     *  - exec        (Esegue frocerie)                 <p>
     *  - ls        (lista dei file)                    <p>
     *  - tree      (albero del progetto)               <p>
     *  - cd        (cambia ambiente di lavoro)         <p>
     *  - load      (carica un immagine)                <p>
     *  - rm        (rimuove un immagine)               <p>
     *  - mkdir     (crea una cartella)                 <p>
     *  - rmdir     (rimuove una cartella)              <p>
     *  - mv        (sposta un immagine o cartella)     <p>
     *  - settings  (apre i settings)                   <p>
     *  - help     (lista dei comandi disponibili)      <p>
     *  - exit     (esce dal progetto)                  <p>
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
        handle_error(RED "Usage:" RESET " exec frocName\n");
    } 
    
    exec(path);

    return 0;
}

int parseLs()
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " ls [path ...]\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto");
    }

    ls(path);

    return 0;
}
int parseTree()
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " tree [path ...]\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto");
    }

    tree(path);

    return 0;
}
int parseCd()
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " cd nameDir\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto");
    }

    cd(path);

    return 0;
}
int parseLoad()
{
    char *path = strtok(nullptr, " ");

    if (path != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " load pathToFile\n");
    }

    // Controlla che l'immagine sia valida
    if (isValidImage(path) == -1)
    {
        handle_error("I formati ammessi sono png/jpeg/ppm");
    }
    loadI(path);

    return 0;
}
int parseRm()
{
    char *path = strtok(nullptr, " ");

    if (path != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " rm filename\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto");
    }

    rm(path);

    return 0;
}
int parseMkdir()
{
    char *name = strtok(nullptr, " ");

    if (name != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " mkdir nameDir\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(name, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto");
    }

    mkdir(name);

    return 0;
}
int parseRmdir()
{
    char *name = strtok(nullptr, " ");

    if (name != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " rmdir nameDir\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(name, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto");
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
        handle_error(RED "Usage:" RESET " mv fromPath toPath\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(pathDst, nullptr);
    if (res != 0)
    {
        handle_error("Il path non si trova all'interno del progetto");
    }
    mv(pathSrc, pathDst);

    return 0;
}
int parseSett(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " settings\n");
    }

    settings(env);

    return 0;
}
int parseHelpP()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " help\n");
    }

    helpP();

    return 0;
}
int parseExitP(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " exit\n");
    }

    exitP(env);

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
        handle_error("Errore nell'esecuzione del comando ls");
    }

    return 0;
}
int tree(char *path)
{
    // Il comando da eseguire
    char comm[256];
    // Salva il comando in comm e controlla che non sia nullptr
    sprintf(comm, "tree %s", path == nullptr ? "." : path);
    // Esegui il comando
    int status = system(comm);
    // Controlla se ci sono errori
    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando tree");
    }
    return 0;
}
int cd(char *path)
{
    // Il comando da eseguire
    char comm[256];
    // TODO: se non viene specificato il path, torni alla $HOME del progetto; va presa da redis
    // Salva il comando in comm
    sprintf(comm, "cd %s", path);
    // Esegui il comando
    int status = system(comm);
    // Controlla se ci sono errori
    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando cd");
    }
    return 0;
}
int loadI(char *path)
{
    //TODO: utilizza mv e cp per caricare l'immagine nella cartella del progetto
    // Il comando da eseguire 
    char comm[256];
    // Controlla che path non sia nullptr
    if (path == nullptr)
    {
        handle_error("Il path non puo' essere null");
    }
    // TODO: Prendi da redis il percorso del progetto, sui cui si dovra' caricare l'immagine
    // Salva il comando
    sprintf(comm, "cp %s %s", path, nullptr);   // nullptr sara' il path del progetto, che verra' caricato da redis
    // Copia l'immagine sul progetto
    int status = system(comm);
    // Controlla se ci sono errori
    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando load");
    }

    return 0;
}
int rm(char *name)
{
    // Il comando da eseguire
    char comm[256];
    // Controlla che name non sia nullptr
    if (name == nullptr)
    {
        handle_error("Il nome del file non puo' essere nullo");
    }
    // Salva il comando in comm
    sprintf(comm, "rm %s", name);
    // Esegui il comando
    int status = system(comm);
    // Controlla se ci sono errori
    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando rm");
    }
    return 0;

}

int mkdir(char *name)
{
    // Il comando da eseguire
    char comm[256];
    // Controlla che name non sia nullptr
    if (name == nullptr)
    {
        handle_error("Il nome della directory non puo' essere nullo");
    }
    // Salva il comando in comm
    sprintf(comm, "mkdir %s", name);
    // Esegui il comando
    int status = system(comm);
    // Controlla se ci sono errori
    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando mkdir");
    }
    return 0;
}

int rmdir(char *name)
{
    // Aggiungi il parametro -r per cancellare ricorsivamente
    sprintf(name, "-r %s", name);
    // Chiama rm
    rm(name);
    return 0;
}

int mv(char *fromPath, char *toPath)
{
    // Il comando da eseguire
    char comm[256];
    // Controlla che fromPath e toPath non siano nullptr
    if (fromPath == nullptr || toPath == nullptr)
    {
        handle_error("Il nome del path non puo' essere nullo");
    }
    // Salva il comando in comm
    sprintf(comm, "mv %s %s", fromPath, toPath);
    // Esegui il comando
    int status = system(comm);
    // Controlla se ci sono errori
    if (status == -1)
    {
        handle_error("Errore nell'esecuzione del comando mv");
    }
    return 0;
}
int settings(Env *env)
{
    //TODO
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
    //TODO:
    D_PRINT("Uscita dal progetto in corso...\n");
    *env = HOMEPAGE;
    return 0;
}


int isPathIn(const char *path, const char *pathProj)
{

    char *absPath = realpath(path, nullptr);
    if (absPath == nullptr)
    { handle_error("Errore nella risoluzione del percorso"); }

    int res = strncmp(absPath, pathProj, strlen(pathProj));

    return res;
}

// Controlla che il file sia un'immagine (png, jpeg, ppm)
int isValidImage(char *path)
{
    // Localizza l'ultima occorrenza del carattere '.'
    char *ext = strrchr(path, '.');
    // Controlla che ci siano errori
    if (ext == nullptr)
    {
        handle_error("Errore nella risoluzione del percorso");
    }
    // Controlla che l'estensione sia valida
    if (strcmp(ext, ".png") != 0 || strcmp(ext, ".jpeg") != 0 || strcmp(ext, ".ppm") != 0)
        return -1;
    return 0;
}