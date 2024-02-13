//
// Created by f3m on 13/02/24.
//

#ifndef EDIX_NEWPGUTILS_HPP
#define EDIX_NEWPGUTILS_HPP

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
int updateSettings(char *projectName, char *tup, int tts, char *tpp, bool backup);


int loadProjectOnRedis(char *projectName);


int delProject(char *name);


int getSettings(char *projectName, char **tup, int *tts, char **tpp, bool *backup);
int getProject(char *projectName, char **cdate, char **mdate, char **path);
int getProjectPath(char *projectName, char **path);


bool checkRoleExists(PGconn *conn, const char *roleName);
bool checkDatabaseExists(PGconn *conn, const char *dbName);


int createQueryImages(char *projectPath, char **names, char **paths, char **query);
char *toExadecimal(unsigned char *imageData, uint width, uint height, uint channels);

#endif //EDIX_NEWPGUTILS_HPP