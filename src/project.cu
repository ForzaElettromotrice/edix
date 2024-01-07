//
// Created by f3m on 07/01/24.
//

#include "project.h"

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
        printf(RED "Usage:" RESET " ls [path]\n");
        return 1;
    }

    //TODO: controlla se il path è dentro alla dir del progetto (per questioni di sicurezza)

    ls(path);

    return 0;
}
int parseTree(char *line)
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        printf(RED "Usage:" RESET " tree [path]\n");
        return 1;
    }

    //TODO: controlla se il path è dentro alla dir del progetto (per questioni di sicurezza)

    tree(path);

    return 0;
}
int parseCd(char *line)
{
    char *path = strtok(nullptr, " ");

    if (strtok(nullptr, " ") != nullptr)
    {
        printf(RED "Usage:" RESET " cd nameDir\n");
        return 1;
    }

    //TODO: controlla se il path è dentro alla dir del progetto (per questioni di sicurezza)

    cd(path);

    return 0;
}
int parseLoad(char *line)
{
    char *path = strtok(nullptr, " ");

    if ( path != nullptr || strtok(nullptr, " ") != nullptr)
    {
        printf(RED "Usage:" RESET " load pathToFile\n");
        return 1;
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
        printf(RED "Usage:" RESET " rm filename\n");
        return 1;
    }

    //TODO: controlla se il path è dentro alla dir del progetto (per questioni di sicurezza)

    rm(path);

    return 0;
}
int parseMkdir(char *line)
{
    char *name = strtok(nullptr, " ");

    if (name != nullptr || strtok(nullptr, " ") != nullptr)
    {
        printf(RED "Usage:" RESET " mkdir dirname\n");
        return 1;
    }

    //TODO: controlla se il name è dentro alla dir del progetto (per questioni di sicurezza)

    mkdir(name);

    return 0;
}
int parseRmdir(char *line)
{
    char *name = strtok(nullptr, " ");

    if (name != nullptr || strtok(nullptr, " ") != nullptr)
    {
        printf(RED "Usage:" RESET " rmdir dirname\n");
        return 1;
    }

    //TODO: controlla se il path è dentro alla dir del progetto (per questioni di sicurezza)

    rmdir(name);

    return 0;
}
int parseMv(char *line)
{
    char *path1 = strtok(nullptr, " ");
    char *path2 = strtok(nullptr, " ");


    if ((path1 != nullptr && path2 != nullptr) || strtok(nullptr, " ") != nullptr)
    {
        printf(RED "Usage:" RESET " mv fromPath toPath\n");
        return 1;
    }

    //TODO: controlla se il path di destinazione è dentro alla dir del progetto (per questioni di sicurezza)

    mv(path1, path2);

    return 0;
}
int parseSett(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        printf(RED "Usage:" RESET " sett\n");
        return 1;
    }

    sett(env);

    return 0;
}
int parseHelpP()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        printf(RED "Usage:" RESET " help\n");
        return 1;
    }

    helpP();

    return 0;
}
int parseExitP(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        printf(RED "Usage:" RESET " exit\n");
        return 1;
    }

    exitP(env);

    return 0;
}


int ls(char *path)
{
    //TODO
    D_PRINT("ls");
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
    //TODO
    D_PRINT("helpP");
    return 0;
}
int exitP(Env *env)
{
    //TODO
    D_PRINT("exitP");
    return 0;
}