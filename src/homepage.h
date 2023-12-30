//
// Created by f3m on 30/12/23.
//

#ifndef EDIX_HOMEPAGE_H
#define EDIX_HOMEPAGE_H

#include <iostream>
#include <cstdarg>
#include "utils.h"


//UTILS
bool isValidName(char *name);
bool isValidFlag(const char *flag);

//PARSERS
int parseHome(char *line, Env *env);
int parseNew(Env *env);
int parseOpen(Env *env);
int parseDel();
int parseView();

//COMMANDS
int newProject(char *name, bool ask);
int openProject(char *name, Env *env);
int delProject(char *name);
int view();

#endif //PHOTOEDITOR_HOMEPAGE_H
