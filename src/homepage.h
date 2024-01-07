//
// Created by f3m on 30/12/23.
//

#ifndef EDIX_HOMEPAGE_H
#define EDIX_HOMEPAGE_H

#include <iostream>
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
int parseHelpH();
int parseExitH(Env *env);

//COMMANDS
int newProject(char *name, bool ask);
int openProject(char *name, Env *env);
int delProject(char *name);
int view();
int helpH();
int exitH(Env *env);

#endif //PHOTOEDITOR_HOMEPAGE_H
