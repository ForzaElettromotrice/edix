//
// Created by f3m on 30/12/23.
//

#ifndef EDIX_HOMEPAGE_HPP
#define EDIX_HOMEPAGE_HPP

#include <iostream>
#include <cstring>
#include <dirent.h>
#include <unistd.h>
#include "project.hpp"
#include "../dbutils/pgutils.hpp"
#include "../utils.hpp"

//UTILS
int banner();
int checkDefaultFolder();
int askParams(char *name, char *path, char *tpp, char *tup, int *tts, bool *backup);
bool isValidName(char *name);
bool isValidFlag(const char *flag);
bool isValidPath(char *path);
bool isValidTPP(char *tpp);
bool isValidTUP(char *tup);
bool isValidTTS(char *tts);
bool isValidBackup(char *backup);

//PARSERS
int parseHome(char *line, Env *env);
int parseNew(Env *env);
int parseOpen(Env *env);
int parseDel();
int parseListH();
int parseHelpH();
int parseExitH(Env *env);

//COMMANDS
int newP(char *name, bool ask, Env *env);
int openP(char *name, Env *env);
int delP(char *name);
int listH();
int helpH();
int exitH(Env *env);

#endif //EDIX_HOMEPAGE_HPP
