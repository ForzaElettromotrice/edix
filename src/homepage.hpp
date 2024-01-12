//
// Created by f3m on 30/12/23.
//

#ifndef EDIX_HOMEPAGE_HPP
#define EDIX_HOMEPAGE_HPP

#include <iostream>
#include <cstring>
#include "dbutils.hpp"
#include "utils.hpp"


//UTILS
bool isValidName(char *name);
bool isValidFlag(const char *flag);
int banner();
int askParams(char *path, char *comp, char *tpp, char *tup, char *modEx, uint *tts, bool *vcs);

//PARSERS
int parseHome(char *line, Env *env);
int parseNew(Env *env);
int parseOpen(Env *env);
int parseDel();
int parseView();
int parseHelpH();
int parseExitH(Env *env);

//COMMANDS
int newP(char *name, bool ask, Env *env);
int openP(char *name, Env *env);
int delP(char *name);
int view();
int helpH();
int exitH(Env *env);

#endif //EDIX_HOMEPAGE_HPP
