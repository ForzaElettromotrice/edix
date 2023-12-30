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
bool isValidFlag(char *flag);

//PARSERS
int parseHome(char *line, Env *env);
int parseNew(Env *env);
int parseOpen(char *line, Env *env);
int parseDel(char *line);
int parseView(char *line);

//COMMANDS
int newProject(char *name, bool ask);

#endif //PHOTOEDITOR_HOMEPAGE_H
