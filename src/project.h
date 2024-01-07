//
// Created by f3m on 07/01/24.
//

#ifndef EDIX_PROJECT_CH
#define EDIX_PROJECT_CH

#include <iostream>
#include "utils.h"

//UTILS

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

#endif //EDIX_PROJECT_CH
