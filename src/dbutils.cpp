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
        system("psql -d postgres -U postgres -c \"SELECT pg_catalog.set_config('search_path', '', false); CREATE ROLE edix NOSUPERUSER CREATEDB NOCREATEROLE INHERIT LOGIN NOREPLICATION NOBYPASSRLS;\" > /dev/null");
    }

    if (!checkDatabaseExists(conn, dbNameToCheck))
        initDb();

    PQfinish(conn);
    return 0;
}
int initDb()
{
    D_PRINT("Creating database...\n");
    system("psql -d postgres -U edix -c \"CREATE DATABASE edix;\" > /dev/null");

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
    D_PRINT("Creating Uint5...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Uint5 AS integer CHECK(VALUE >= 5);\" > /dev/null");

    D_PRINT("Creating table Project...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Project (Name VARCHAR(50) PRIMARY KEY NOT NULL,CDate TIMESTAMP NOT NULL,MDate TIMESTAMP NOT NULL,Path VARCHAR(256) UNIQUE NOT NULL,Settings INT NOT NULL);\" > /dev/null");
    D_PRINT("Creating table Settings...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Settings (Id SERIAL PRIMARY KEY NOT NULL,TUP Tupx NOT NULL,Mod_ex Modex NOT NULL,Comp Compx NOT NULL,TTS Uint5 NOT NULL,TPP Tppx NOT NULL,VCS BOOLEAN NOT NULL,Project VARCHAR(50) NOT NULL);\" > /dev/null");
    D_PRINT("Creating table Dix...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Dix (Instant TIMESTAMP PRIMARY KEY NOT NULL,Project VARCHAR(50) NOT NULL, Name VARCHAR(50) NOT NULL, Comment VARCHAR(1024),UNIQUE (Project, Name),CONSTRAINT V6 FOREIGN KEY (Project) REFERENCES Project(Name) ON DELETE CASCADE);\" > /dev/null");
    D_PRINT("Creating table Photo...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Photo (Id SERIAL PRIMARY KEY NOT NULL,Name VARCHAR(50) NOT NULL,Path VARCHAR(256) NOT NULL,Comp Compx NOT NULL,Project VARCHAR(50),Dix TIMESTAMP,CONSTRAINT V7 FOREIGN KEY (Project) REFERENCES Project(Name) ON DELETE CASCADE,CONSTRAINT V8 FOREIGN KEY (Dix) REFERENCES Dix(Instant) ON DELETE CASCADE,CONSTRAINT V9 CHECK ((Project IS NOT NULL AND Dix IS NULL) OR (Project IS NULL AND Dix IS NOT NULL)));\" > /dev/null");

    D_PRINT("Altering table Project...\n");
    system("psql -d edix -U edix -c \"ALTER TABLE Project ADD CONSTRAINT V1 CHECK (CDate <= MDate),ADD CONSTRAINT V2 UNIQUE (Settings),ADD CONSTRAINT V3 FOREIGN KEY (Settings) REFERENCES Settings(Id) ON DELETE CASCADE INITIALLY DEFERRED;\" > /dev/null");
    D_PRINT("Altering table Settings...\n");
    system("psql -d edix -U edix -c \"ALTER TABLE Settings ADD CONSTRAINT V4 UNIQUE (Project),ADD CONSTRAINT V5 FOREIGN KEY (Project) REFERENCES Project(Name) ON DELETE CASCADE INITIALLY DEFERRED;\" > /dev/null");

    D_PRINT("Creating function CheckTimeFunction...\n");
    system(R"(psql -d edix -U edix -c "CREATE FUNCTION CheckTimeFunction() RETURNS TRIGGER AS \$\$ BEGIN IF NEW.Instant < (SELECT CDate FROM Project p WHERE p.name = NEW.Project) THEN RAISE EXCEPTION 'Cannot insert a dix with Instant < CDate'; END IF; RETURN NEW; END; \$\$ LANGUAGE plpgsql;" > /dev/null)");

    D_PRINT("Creating trigger on Dix...\n");
    system("psql -d edix -U edix -c \"CREATE TRIGGER CheckTime BEFORE INSERT ON Dix FOR EACH ROW EXECUTE FUNCTION CheckTimeFunction();\" > /dev/null");

    return 0;
}
int checkPostgresService()
{
    PGconn *conn = PQconnectdb("dbname=postgres user=postgres password=");

    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        handle_error("Postgres is not running\n");
    }

    return 0;
}

int loadProjectOnRedis(char *projectName)
{

    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        handle_error("(Errore di connessione), %s\n", PQerrorMessage(conn));
    }


    char **settings = getSettings(conn, projectName);
    if (settings == nullptr)
        return 1;


    settingsToRedis((int) strtol(settings[0], nullptr, 10),
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
int addProject(char *name, char *path, char *comp, char *TPP, char *TUP, char *modEx, uint TTS, bool VCS)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        handle_error("Errore di connessione %s\n", PQerrorMessage(conn));
    }

    char query[1024];
    sprintf(query, "BEGIN;\n"
                   "DO $$\n"
                   "DECLARE\n"
                   "settingsId INT;\n"
                   "BEGIN\n"
                   "INSERT INTO Project (Name, CDate, MDate, Path, Settings) VALUES ('%s', NOW(), NOW(), '%s', -1);\n"
                   "INSERT INTO Settings (TUP, Mod_ex, Comp, TTS, TPP, VCS, Project) VALUES ('%s', '%s', '%s', %d, '%s', %s, '%s') RETURNING Id INTO settingsId;\n"
                   "UPDATE Project SET Settings = settingsId WHERE Name = '%s';\n"
                   "END $$;\n"
                   "COMMIT;\n", name, path, TUP, modEx, comp, TTS, TPP, VCS ? "TRUE" : "FALSE", name, name);
    D_PRINT("Adding project to Postgres...\n");
    PGresult *res = PQexec(conn, query);

    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        const char *sqlstate = PQresultErrorField(res, PG_DIAG_SQLSTATE);

        if (strcmp(sqlstate, "23505") == 0)
            fprintf(stderr, RED "Error:" RESET " Questo progetto già esiste!\n");
        else
            fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQerrorMessage(conn));


        PQclear(res);
        PQfinish(conn);
        return 1;
    }


    PQclear(res);
    PQfinish(conn);
    return 0;
}
int addDix(char *projectName, char *name, char *comment)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        handle_error("Errore di connessione %s\n", PQerrorMessage(conn));
    }

    char query[1024];
    sprintf(query, "INSERT INTO Dix (Instant, Name, Comment, Project) VALUES (NOW(), '%s', '%s', '%s');\n", name,
            comment, projectName);

    D_PRINT("Adding project to Postgres...\n");
    PGresult *res = PQexec(conn, query);

    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        const char *sqlstate = PQresultErrorField(res, PG_DIAG_SQLSTATE);

        if (strcmp(sqlstate, "23505") == 0)
            fprintf(stderr, RED "Error:" RESET " Questo dix già esiste!\n");
        else
            fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQerrorMessage(conn));


        PQclear(res);
        PQfinish(conn);
        return 1;
    }


    PQclear(res);
    PQfinish(conn);
    return 0;
}

int delProject(char *name)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        handle_error("Errore di connessione %s\n", PQerrorMessage(conn));
    }

    char *path = getPath(conn, name);
    if (path == nullptr)
    {
        PQfinish(conn);
        return 1;
    }

    D_PRINT("Removing project folder...\n");
    char command[256];
    sprintf(command, "rm -rf %s", path);
    if (system(command) != 0)
    {
        PQfinish(conn);
        handle_error("Impossibile rimuovere cartella del progetto\n");
    }

    char query[256];
    sprintf(query, "DELETE FROM Project p WHERE p.name = '%s';", name);

    D_PRINT("Removing project from Postgres...\n");
    PGresult *res = PQexec(conn, query);

    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        PQclear(res);
        PQfinish(conn);
        handle_error("Errore nell'esecuzione della query -> %s", PQerrorMessage(conn));
    }
    PQclear(res);


    PQfinish(conn);

    return 0;
}


char *getProjects()
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");


    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        fprintf(stderr, "Errore di connessione: %s\n", PQerrorMessage(conn));
        return nullptr;
    }


    char query[256];
    sprintf(query, "SELECT Name FROM Project");

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return nullptr;
    }


    int numRows = PQntuples(result);
    int numCols = PQnfields(result);

    char *names = (char *) malloc(1024 * sizeof(char));
    sprintf(names, BLUE BOLD "Progetti:\n" RESET);

    for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numCols; j++)
        {
            strcat(names, YELLOW "- ");
            strcat(names, PQgetvalue(result, i, j));
            strcat(names, "\n" RESET);
        }

    PQclear(result);
    PQfinish(conn);
    return names;
}
char *getDixs(char *projectName)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");


    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        fprintf(stderr, "Errore di connessione: %s\n", PQerrorMessage(conn));
        return nullptr;
    }


    char query[256];
    sprintf(query, "SELECT Name, Instant, Comment FROM Dix d WHERE d.Project = '%s' ORDER BY Instant", projectName);

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return nullptr;
    }


    int numRows = PQntuples(result);
    int numCols = PQnfields(result);

    char *names = (char *) malloc(1024 * sizeof(char));
    sprintf(names, BLUE BOLD "Dix su %s:\n" RESET, projectName);

    for (int i = 0; i < numRows; i++)
    {
        strcat(names, YELLOW "- ");
        for (int j = 0; j < numCols; j++)
        {
            strcat(names, PQgetvalue(result, i, j));
            strcat(names, "\t");
        }
        strcat(names, "\n" RESET);
    }

    PQclear(result);
    PQfinish(conn);
    return names;
}
char **getSettings(PGconn *conn, char *projectName)
{
    char query[256];
    sprintf(query, "SELECT * FROM Settings WHERE Project = '%s'", projectName);

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
            values[j] = strdup(PQgetvalue(result, i, j));


    values[numCols] = nullptr;
    PQclear(result);

    return values;
}
char *getPath(PGconn *conn, char *name)
{

    char query[256];
    sprintf(query, "SELECT path FROM Project p WHERE p.name = '%s'", name);

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return nullptr;
    }

    char *path = strdup(PQgetvalue(result, 0, 0));

    PQclear(result);
    return path;
}
bool existProject(char *name)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");


    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        fprintf(stderr, "Errore di connessione: %s\n", PQerrorMessage(conn));
        return false;
    }


    char query[256];
    sprintf(query, "SELECT Name FROM Project p Where p.Name = '%s' ", name);

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return false;
    }


    int numRows = PQntuples(result);


    PQclear(result);
    PQfinish(conn);
    return numRows == 1;
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