//
// Created by f3m on 12/01/24.
//

#ifndef EDIX_DBUTILS_HPP
#define EDIX_DBUTILS_HPP

#include <libpq-fe.h>
#include <hiredis/hiredis.h>
#include <cstdlib>
#include <cstring>
#include "rdutils.hpp"
#include "utils.hpp"


int initDb();
int checkDb();
int loadProjectOnRedis(char *projectName);
int addProject(char *name, char *path, char *comp, char *TPP, char *TUP, char *modEx, uint TTS, bool VCS);


//UTILS
char **getSettings(PGconn *conn, char *projectName);
char *getProjects();

bool checkRoleExists(PGconn *conn, const char *roleName);
bool checkDatabaseExists(PGconn *conn, const char *dbName);


#endif //EDIX_DBUTILS_HPP
