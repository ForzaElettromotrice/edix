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
#include "../utils.hpp"


int initDb();
int checkDb();
int checkPostgresService();

int loadProjectOnRedis(char *projectName);
int loadDix(char *name, char *projectName);
int addProject(char *name, char *path, char *comp, char *TPP, char *TUP, char *modEx, uint TTS, bool Backup);
int addDix(char *projectName, char *dixName, char *comment, char **images, char **paths);
int delProject(char *name);
int updateSettings(int id, char *tup, char *mod_ex, char *comp, u_int tts, char *tpp, bool backup, char *pName);

//UTILS
char *getProjects();
char *getDixs(char *projectName);
char **getProject(PGconn *conn, char *projectName);
char **getSettings(PGconn *conn, char *projectName);
char *getPath(PGconn *conn, char *name);
unsigned char *getImageData(char *path, size_t *dim);
bool existProject(char *name);

bool checkRoleExists(PGconn *conn, const char *roleName);
bool checkDatabaseExists(PGconn *conn, const char *dbName);
void changeFormat(char **comment);

#endif //EDIX_DBUTILS_HPP
