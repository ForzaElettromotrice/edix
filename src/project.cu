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
    //TODO
    return 0;
}
int parseTree(char *line)
{
    //TODO
    return 0;
}
int parseCd(char *line)
{
    //TODO
    return 0;
}
int parseLoad(char *line)
{
    //TODO
    return 0;
}
int parseRm(char *line)
{
    //TODO
    return 0;
}
int parseMkdir(char *line)
{
    //TODO
    return 0;
}
int parseRmdir(char *line)
{
    //TODO
    return 0;
}
int parseMv(char *line)
{
    //TODO
    return 0;
}
int parseSett(Env *env)
{
    //TODO
    return 0;
}
int parseHelpP()
{
    //TODO
    return 0;
}
int parseExitP(Env *env)
{
    //TODO
    return 0;
}