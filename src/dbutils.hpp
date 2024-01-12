//
// Created by f3m on 12/01/24.
//

#ifndef EDIX_DBUTILS_HPP
#define EDIX_DBUTILS_HPP

#include <libpq-fe.h>

#include <cstdlib>
#include "utils.hpp"


//UTILS
bool checkRoleExists(PGconn *conn, const char *roleName);
bool checkDatabaseExists(PGconn *conn, const char *dbName);


int initDb();
int checkDb();

#endif //EDIX_DBUTILS_HPP
