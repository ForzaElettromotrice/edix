//
// Created by f3m on 12/01/24.
//

#ifndef EDIX_DBUTILS_HPP
#define EDIX_DBUTILS_HPP

#include <libpq-fe.h>
#include <hiredis/hiredis.h>
#include <cstdlib>
#include <cstring>
#include "utils.hpp"


int initDb();
int checkDb();
int loadProjectOnRedis(char *projectName);
int upload_to_redis(int id, char *tup, char *mod_ex, char *comp, unsigned int tts, char *tpp, bool vcs, int project);
int addProject(char *name, char *path, char *comp, char *TPP, char *TUP, char *modEx, uint TTS, bool VCS);


//UTILS
int get_project_id(char *projectName, char **ID);
int get_check(char *name, redisReply *reply, redisContext *context);
int set_check(char *name, redisReply *reply, redisContext *context);
char **get_settings(PGconn *conn, char *projectId);

bool checkRoleExists(PGconn *conn, const char *roleName);
bool checkDatabaseExists(PGconn *conn, const char *dbName);


#endif //EDIX_DBUTILS_HPP
