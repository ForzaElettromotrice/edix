//
// Created by f3m on 07/01/24.
//

#ifndef EDIX_PROJECT_CH
#define EDIX_PROJECT_CH

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include "dbutils.hpp"
#include "utils.hpp"

//UTILS
int isPathIn(const char *path, const char *pathProj);
int isValidImage(char *path);
char *askComment();
int checkDix(char *projectPath, char *dixName);
int cloneProject(char *projectPath, char *path, char *dixName);


//PARSERS
int parseProj(char *line, Env *env);
int parseLs();
int parseExec();
int parseTree();
int parseCd();
int parseLoad();
int parseRm();
int parseMkdir();
int parseRmdir();
int parseMv();
int parseSett(Env *env);
int parseDix();
int parseForce();
int parseHelpP();
int parseExitP(Env *env);


//COMMANDS
int ls(const char *path);
int exec(char *path);
int tree(const char *path);
int cd(char *path);
int loadI(char *path);
int rm(char *name);
int mkdir(char *name);
int rmdir(char *name);
int mv(char *fromPath, char *toPath);
int settings(Env *env);
int helpP();
int exitP(Env *env);
int dixCommit(char *name);
int dixList();
int dixReload(char *name);
int force();


#endif //EDIX_PROJECT_CH
