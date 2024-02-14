//
// Created by f3m on 13/02/24.
//

#ifndef EDIX_PGUTILS_HPP
#define EDIX_PGUTILS_HPP

#include <libpq-fe.h>
#include <cstring>
#include "rdutils.hpp"
#include "omp.h"
#include "../functions/imgutils.hpp"
#include "../utils.hpp"


int initDb();
int checkDb();
int checkPostgresService();


int addProject(char *name, char *path, char *tup, int tts, char *tpp, bool backup);
int addDix(char *dixName, char *comment, char **images, char **paths);
int updateSettings(char *projectName, char *tup, int tts, char *tpp, bool backup);


int loadProjectOnRedis(char *projectName);
int loadDix(char *dixName, char *projectName);


int delProject(char *name);


char *listDixs(char *projectName);
char *listProjects();


int getSettings(char *projectName, char **tup, int *tts, char **tpp, bool *backup);
int getProject(char *projectName, char **cdate, char **mdate, char **path);
int getProjectPath(char *projectName, char **path);


bool checkRoleExists(PGconn *conn, const char *roleName);
bool checkDatabaseExists(PGconn *conn, const char *dbName);
bool existProject(char *name);


int createQueryImages(char *projectPath, char *dixName, char **names, char **paths, char **query);
char *toExadecimal(unsigned char *imageData, uint width, uint height, uint channels);
unsigned char *toImg(char *exaData, uint width, uint height, uint channels);
int adjustComment(char **comment, char *line, uint padding);
int buildPath(char *path);

#endif //EDIX_PGUTILS_HPP