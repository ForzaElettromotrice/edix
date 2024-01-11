//
// Created by f3m on 07/01/24.
//

#ifndef EDIX_PROJECT_CH
#define EDIX_PROJECT_CH

#include <iostream>
#include "utils.h"
#include <stdlib.h>

//UTILS
int isPathIn(const char *, const char *);

//PARSERS
int parseProj(char *line, Env *env);
int parseLs(char *line);
int parseTree(char *line);
int parseCd(char *line);
int parseLoad(char *line);
int parseRm(char *line);
int parseMkdir(char *line);
int parseRmdir(char *line);
int parseMv(char *line);
int parseSett(Env *env);
int parseHelpP();
int parseExitP(Env *env);


//COMMANDS
int ls(char *path);
int tree(char *path);
int cd(char *path);
int loadI(char *path);
int rm(char *name);
int mkdir(char *name);
int rmdir(char *name);
int mv(char *fromPath, char *toPath);
int settings(Env *env);
int helpP();
int exitP(Env *env);


#endif //EDIX_PROJECT_CH
