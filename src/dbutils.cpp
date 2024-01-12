//
// Created by f3m on 12/01/24.
//

#include "dbutils.hpp"


int checkDb()
{
    PGconn *conn = PQconnectdb("dbname=postgres user=postgres password=");

    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        fprintf(stderr, "Errore di connessione a PostgreSQL: %s\n", PQerrorMessage(conn));
        return 1;
    }
    const char *roleNameToCheck = "edix";
    const char *dbNameToCheck = "edix";

    if (!checkRoleExists(conn, roleNameToCheck))
    {
        D_PRINT("Creating user edix...\n");
        system("createuser edix -d > /dev/null");
    }

    if (!checkDatabaseExists(conn, dbNameToCheck))
        initDb();

    PQfinish(conn);
    return 0;
}
int loadProjectOnRedis(char *projectName)
{
    // Crea una connessione al database
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");

    // Verifica lo stato della connessione
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        fprintf(stderr, "Errore di connessione: %s\n", PQerrorMessage(conn));
        return 1;
    }

    char *projectId;
    get_project_id(projectName, &projectId);


    char **settings = get_settings(conn, projectId);
    if (settings == nullptr)
        return 1;


    upload_to_redis((int) strtol(settings[0], nullptr, 10),
                    settings[1],
                    settings[2],
                    settings[3],
                    strtoul(settings[4], nullptr, 10),
                    settings[5],
                    strcmp("t", settings[6]) == 0,
                    (int) strtol(settings[7], nullptr, 10));


    for (int i = 0; settings[i] != nullptr; ++i)
        free(settings[i]);
    free(settings);
    PQfinish(conn);

    return 0;
}

int initDb()
{
    D_PRINT("Creating database...\n");
    system("createdb edix -O edix > /dev/null");

    D_PRINT("Creating Tupx_t...\n");
    system("psql -d edix -U edix -c \"CREATE TYPE Tupx_t AS ENUM ('Bilinear', 'Bicubic');\" > /dev/null");
    D_PRINT("Creating Modex_t...\n");
    system("psql -d edix -U edix -c \"CREATE TYPE Modex_t AS ENUM('Immediate', 'Programmed');\" > /dev/null");
    D_PRINT("Creating Compx_t...\n");
    system("psql -d edix -U edix -c \"CREATE TYPE Compx_t AS ENUM('JPEG', 'PNG', 'PPM');\" > /dev/null");
    D_PRINT("Creating Tppx_t...\n");
    system("psql -d edix -U edix -c \"CREATE TYPE Tppx_t AS ENUM('Serial', 'OMP', 'CUDA');\" > /dev/null");

    D_PRINT("Creating Tupx...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Tupx AS Tupx_t;\" > /dev/null");
    D_PRINT("Creating Modex...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Modex AS Modex_t;\" > /dev/null");
    D_PRINT("Creating Compx...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Compx AS Compx_t;\" > /dev/null");
    D_PRINT("Creating Tppx...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Tppx AS Tppx_t;\" > /dev/null");
    D_PRINT("Creating Uint...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Uint AS integer CHECK(VALUE >= 0);\" > /dev/null");

    D_PRINT("Creating table Project...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Project (Id SERIAL PRIMARY KEY NOT NULL,Name VARCHAR(50) NOT NULL,CDate TIMESTAMP NOT NULL,MDate TIMESTAMP NOT NULL,Path VARCHAR(256) UNIQUE NOT NULL,Settings INT NOT NULL);\" > /dev/null");
    D_PRINT("Creating table Settings_p...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Settings (Id SERIAL PRIMARY KEY NOT NULL,TUP Tupx NOT NULL,Mod_ex Modex NOT NULL,Comp Compx NOT NULL,TTS INT NOT NULL,TPP Tppx NOT NULL,VCS BOOLEAN NOT NULL,Project INT NOT NULL);\" > /dev/null");

    D_PRINT("Altering table Project...\n");
    system("psql -d edix -U edix -c \"ALTER TABLE Project ADD CONSTRAINT V1 CHECK (CDate <= MDate),ADD CONSTRAINT V2 UNIQUE (Settings),ADD CONSTRAINT V3 FOREIGN KEY (Settings) REFERENCES Settings(Id) INITIALLY DEFERRED;\" > /dev/null");
    D_PRINT("Altering table Settings_p...\n");
    system("psql -d edix -U edix -c \"ALTER TABLE Settings ADD CONSTRAINT V4 UNIQUE (Project),ADD CONSTRAINT V5 FOREIGN KEY (Project) REFERENCES Project(Id) INITIALLY DEFERRED;\" > /dev/null");

    D_PRINT("Creating table Dix...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Dix (Instant TIMESTAMP PRIMARY KEY NOT NULL,Project INT NOT NULL,CONSTRAINT V6 FOREIGN KEY (Project) REFERENCES Project(Id));\" > /dev/null");
    D_PRINT("Creating table Photo...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Photo (Id SERIAL PRIMARY KEY NOT NULL,Name VARCHAR(50) NOT NULL,Path VARCHAR(256) NOT NULL,Comp Compx NOT NULL,Project INT,Dix TIMESTAMP,CONSTRAINT V7 FOREIGN KEY (Project) REFERENCES Project(Id),CONSTRAINT V8 FOREIGN KEY (Dix) REFERENCES Dix(Instant),CONSTRAINT V9 CHECK ((Project IS NOT NULL AND Dix IS NULL) OR (Project IS NULL AND Dix IS NOT NULL)));\" > /dev/null");


    return 0;
}


int get_check(char *name, redisReply *reply, redisContext *context)
{
    if (reply == nullptr)
    {
        printf("Errore nell'esecuzione del comando GET %s\n", name);
        freeReplyObject(reply);
        redisFree(context);
        return 1;
    }

    // Visualizza i dati recuperati
    printf("Dati recuperati: %s\n", reply->str);

    freeReplyObject(reply);
    // ????
    redisFree(context);
    return 0;
}
int set_check(char *name, redisReply *reply, redisContext *context)
{
    if (reply == nullptr)
    {
        printf("Errore nell'esecuzione del comando SET %s\n", name);
        freeReplyObject(reply);
        redisFree(context);
        return 1;
    }

    freeReplyObject(reply);
    return 0;
}

int get_project_id(char *projectName, char **ID)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");

    // Verifica lo stato della connessione
    if (PQstatus(conn) != CONNECTION_OK)
    {
        fprintf(stderr, "Errore di connessione: %s\n", PQerrorMessage(conn));
        PQfinish(conn);
        return 1;
    }

    // Esegui una query SQL
    char query[256];
    sprintf(query, "SELECT ID FROM Project WHERE Name = '%s'", projectName);

    PGresult *result = PQexec(conn, query);

    // Verifica lo stato della query
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        PQclear(result);
        PQfinish(conn);
        return 1;
    }

    // Recupera e stampa i risultati
    int numRows = PQntuples(result);
    int numCols = PQnfields(result);

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            *ID = PQgetvalue(result, i, j);
        }
    }

    //Libera la memoria delle risorse
    PQclear(result);
    PQfinish(conn);

    return 0;
}
char **get_settings(PGconn *conn, char *projectId)
{
    char query[256];
    sprintf(query, "SELECT * FROM Settings WHERE Project = %s", projectId);

    PGresult *result = PQexec(conn, query);


    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        PQclear(result);
        PQfinish(conn);
        return nullptr;
    }


    int numRows = PQntuples(result);
    int numCols = PQnfields(result);


    char **values = (char **) malloc((numCols + 1) * sizeof(char *));

    for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numCols; j++)
            values[j] = PQgetvalue(result, i, j);


    values[numCols] = nullptr;
    PQclear(result);

    return values;
}
int upload_to_redis(int id, char *tup, char *mod_ex, char *comp, u_int tts, char *tpp, bool vcs, int project)
{


    // Connetti a Redis
    redisContext *context = redisConnect("localhost", 6379);

    if (context == nullptr || context->err)
    {
        if (context)
        {
            printf("Errore di connessione a Redis: %s\n", context->errstr);
            redisFree(context);
            return 1;
        } else
        {
            printf("Impossibile inizializzare la connessione a Redis\n");
            return 1;
        }
    }

    //Esegui i comandi per caricare le settings su redis
    auto *reply = (redisReply *) redisCommand(context, "SET ID %d", id);
    set_check((char *) "ID", reply, context);


    reply = (redisReply *) redisCommand(context, "SET Project %d", project);
    set_check((char *) "Project", reply, context);


    reply = (redisReply *) redisCommand(context, "SET Mod_ex %s", mod_ex);
    set_check((char *) "Mod_ex", reply, context);

    reply = (redisReply *) redisCommand(context, "SET TTS %u", tts);
    set_check((char *) "TTS", reply, context);

    reply = (redisReply *) redisCommand(context, "SET VCS %d", vcs);
    set_check((char *) "VCS", reply, context);


    reply = (redisReply *) redisCommand(context, "SET COMP %s", comp);
    set_check((char *) "COMP", reply, context);

    reply = (redisReply *) redisCommand(context, "SET TPP %s", tpp);
    set_check((char *) "TPP", reply, context);

    reply = (redisReply *) redisCommand(context, "SET TUP %s", tup);
    set_check((char *) "TUP", reply, context);


    // Recupera i dati dalle chiavi
    reply = (redisReply *) redisCommand(context, "GET %s", "ID");
    get_check((char *) "ID", reply, context);

    reply = (redisReply *) redisCommand(context, "GET %s", "Project");
    get_check((char *) "Project", reply, context);

    reply = (redisReply *) redisCommand(context, "GET %s", "Mod_ex");
    get_check((char *) "Mod_ex", reply, context);

    reply = (redisReply *) redisCommand(context, "GET %s", "TTS");
    get_check((char *) "TTS", reply, context);

    reply = (redisReply *) redisCommand(context, "GET %s", "VCS");
    get_check((char *) "VCS", reply, context);

    reply = (redisReply *) redisCommand(context, "GET %s", "COMP");
    get_check((char *) "COMP", reply, context);

    reply = (redisReply *) redisCommand(context, "GET %s", "TPP");
    get_check((char *) "TPP", reply, context);

    reply = (redisReply *) redisCommand(context, "GET %s", "TUP");
    get_check((char *) "TUP", reply, context);

    return 0;

}


bool checkRoleExists(PGconn *conn, const char *roleName)
{
    char query[256];
    sprintf(query, "SELECT 1 FROM pg_roles WHERE rolname = '%s'", roleName);
    PGresult *res = PQexec(conn, query);

    bool out = (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0);
    PQclear(res);

    return out;
}
bool checkDatabaseExists(PGconn *conn, const char *dbName)
{
    char query[256];
    sprintf(query, "SELECT 1 FROM pg_database WHERE datname = '%s'", dbName);
    PGresult *res = PQexec(conn, query);

    bool out = (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0);
    PQclear(res);

    return out;
}