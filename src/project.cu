//
// Created by f3m on 07/01/24.
//

#include "project.h"
#include <stdlib.h>

int parseProj(char *line, Env *env)
{
    /**
     *  11 comandi                                  <p>
     *  - ls        (lista dei file)                <p>
     *  - tree      (albero del progetto)           <p>
     *  - cd        (cambia ambiente di lavoro)     <p>
     *  - load      (carica un immagine)            <p>
     *  - rm        (rimuove un immagine)           <p>
     *  - mkdir     (crea una cartella)             <p>
     *  - rmdir     (rimuove una cartella)          <p>
     *  - mv        (sposta un immagine o cartella) <p>
     *  - settings  (apre i settings)               <p>
     *  - helpH     (lista dei comandi disponibili) <p>
     *  - exitH     (esce dal progetto)             <p>
     */
    char *copy = strdup(line);
    char *token = strtok(copy, " ");


    if (strcmp(token, "ls") == 0)
        parseLs(line);
    else if (strcmp(token, "tree") == 0)
        parseTree(line);
    else if (strcmp(token, "cd") == 0)
        parseCd(line);
    else if (strcmp(token, "load") == 0)
        parseLoad(line);
    else if (strcmp(token, "rm") == 0)
        parseRm(line);
    else if (strcmp(token, "mkdir") == 0)
        parseMkdir(line);
    else if (strcmp(token, "rmdir") == 0)
        parseRmdir(line);
    else if (strcmp(token, "mv") == 0)
        parseMv(line);
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

int parseLs(char *line)
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " ls [path]\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0) {
        handle_error("Il path non si trova all'interno del progetto");
    }

    ls(path);

    return 0;
}
int parseTree(char *line)
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " tree [path]\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0) {
        handle_error("Il path non si trova all'interno del progetto");
    }

    tree(path);

    return 0;
}
int parseCd(char *line)
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " cd nameDir\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0) {
        handle_error("Il path non si trova all'interno del progetto");
    }

    cd(path);

    return 0;
}
int parseLoad(char *line)
{
    char *path = strtok(nullptr, " ");

    if ( path != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " load pathToFile\n");
    }

    //TODO: controlla se il file è un png/jpeg/ppm

    loadI(path);

    return 0;
}
int parseRm(char *line)
{
    char *path = strtok(nullptr, " ");

    if ( path != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " rm filename\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(path, nullptr);
    if (res != 0) {
        handle_error("Il path non si trova all'interno del progetto");
    }

    rm(path);

    return 0;
}
int parseMkdir(char *line)
{
    char *name = strtok(nullptr, " ");

    if (name != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " mkdir dirname\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(name, nullptr);
    if (res != 0) {
        handle_error("Il path non si trova all'interno del progetto");
    }

    mkdir(name);

    return 0;
}
int parseRmdir(char *line)
{
    char *name = strtok(nullptr, " ");

    if (name != nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " rmdir dirname\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(name, nullptr);
    if (res != 0) {
        handle_error("Il path non si trova all'interno del progetto");
    }

    rmdir(name);

    return 0;
}
int parseMv(char *line)
{
    char *pathSrc = strtok(nullptr, " ");
    char *pathDst = strtok(nullptr, " ");


    if ((pathSrc != nullptr && pathDst != nullptr) || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " mv fromPath toPath\n");
    }

    //TODO: Al posto di nullptr, va il path del progetto andra' preso da redis
    int res = isPathIn(pathDst, nullptr);
    if (res != 0) {
        handle_error("Il path non si trova all'interno del progetto");
    }
    mv(pathSrc, pathDst);

    return 0;
}
int parseSett(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " sett\n");
    }

    sett(env);

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


int ls(char *path)
{
    // Il comando da eseguire
    char comm[256];
    // Controlla che path non sia nullptr
    if (path == nullptr) { path = "."; }
    // Salva il comando in comm
    sprintf(comm, "ls %s", path);
    // Esegui il comando
    int status = system(comm);
    // Controlla se ci sono errori
    if (status == -1) {
        handle_error("Errore nell'esecuzione del comando ls");
    }
    return 0;
}
int tree(char *path)
{
    //TODO
    D_PRINT("tree");
    return 0;
}
int cd(char *path)
{
    //TODO
    D_PRINT("cd");
    return 0;
}
int loadI(char *path)
{
    //TODO
    D_PRINT("loadI");
    return 0;
}
int rm(char *name)
{
    //TODO
    D_PRINT("rm");
    return 0;
}
int mkdir(char *name)
{
    //TODO
    D_PRINT("mkdir");
    return 0;
}
int rmdir(char *name)
{
    //TODO
    D_PRINT("rmdir");
    return 0;
}
int mv(char *fromPath, char *toPath)
{
    //TODO
    D_PRINT("mv");
    return 0;
}
int sett(Env *env)
{
    //TODO
    D_PRINT("sett");
    return 0;
}
int helpP()
{
    D_PRINT("Ecco la lista dei comandi che puoi utilizzare all'interno del tuo progetto:\n"
            BOLD "  ls" RESET "\tStampa il contenuto della directory\n"
            BOLD "  tree" RESET "\tStampa il contenuto della directory in un formato ad albero\n"
            BOLD "  cd" RESET "\tCambia directory\n"
            BOLD "  loadI" RESET "\tCarica un'immagine\n"
            BOLD "  rm" RESET "\tRimuovi un file\n"
            BOLD "  mkdir" RESET "\tCrea una directory\n"
            BOLD "  rmdir" RESET "\tRimuovi una directory\n"
            BOLD "  mv" RESET "\tSposta un file o lo rinomina\n"
            BOLD "  sett" RESET "\tAccedi ai setting\n");
    return 0;
}
int exitP(Env *env)
{
    //TODO
    D_PRINT("exitP");
    return 0;
}

// Controlla che path sia all'interno di pathProj
int isPathIn(const char *path, const char *pathProj) {
    // Prendi il percorso assoluto di path
    char *absPath = realpath(path, nullptr);
    // Controlla che ci siano errori
    if (absPath == nullptr)
    {
        handle_error("Errore nella risoluzione del percorso");
    }
    // Confrontalo con il path del progetto
    int res = strncmp(absPath, pathProj, strlen(pathProj));

    return res;
}