//
// Created by f3m on 13/02/24.
//

#include "pgutils.hpp"

int checkDb()
{
    PGconn *conn = PQconnectdb("dbname=postgres user=postgres password=");

    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print(RED "POSTGRES Error :" RESET "di connessione a PostgreSQL: %s\n", PQerrorMessage(conn));
        return 1;
    }
    const char *roleNameToCheck = "edix";
    const char *dbNameToCheck = "edix";

    if (!checkRoleExists(conn, roleNameToCheck))
    {
        D_Print("Creating user edix...\n");
        system("psql -d postgres -U postgres -c \"SELECT pg_catalog.set_config('search_path', '', false); CREATE ROLE edix NOSUPERUSER CREATEDB NOCREATEROLE INHERIT LOGIN NOREPLICATION NOBYPASSRLS;\" > /dev/null");
    }

    if (!checkDatabaseExists(conn, dbNameToCheck))
        initDb();

    PQfinish(conn);
    return 0;
}
int initDb()
{
    D_Print("Creating database...\n");
    system("psql -d postgres -U edix -c \"CREATE DATABASE edix;\" > /dev/null");

    D_Print("Creating Tupx_t...\n");
    system("psql -d edix -U edix -c \"CREATE TYPE Tupx_t AS ENUM ('Bilinear', 'Bicubic');\" > /dev/null");
    D_Print("Creating Tppx_t...\n");
    system("psql -d edix -U edix -c \"CREATE TYPE Tppx_t AS ENUM('Serial', 'OMP', 'CUDA');\" > /dev/null");

    D_Print("Creating Tupx...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Tupx AS Tupx_t;\" > /dev/null");
    D_Print("Creating Tppx...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Tppx AS Tppx_t;\" > /dev/null");
    D_Print("Creating Uint5...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Uint5 AS integer CHECK(VALUE >= 5);\" > /dev/null");
    D_Print("Creating Uint...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Uint AS integer CHECK(VALUE >= 0);\" > /dev/null");
    D_Print("Creating Cint...\n");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Cint AS integer CHECK(VALUE >= 1 AND VALUE <= 3);\" > /dev/null");

    D_Print("Creating table Project...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Project (Name VARCHAR(50) PRIMARY KEY NOT NULL,CDate TIMESTAMP NOT NULL,MDate TIMESTAMP NOT NULL,Path VARCHAR(256) UNIQUE NOT NULL,Settings INT NOT NULL);\" > /dev/null");
    D_Print("Creating table Settings...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Settings (Id SERIAL PRIMARY KEY NOT NULL,TUP Tupx NOT NULL,TTS Uint5 NOT NULL,TPP Tppx NOT NULL,Backup BOOLEAN NOT NULL,Project VARCHAR(50) NOT NULL);\" > /dev/null");
    D_Print("Creating table Dix...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Dix (Instant TIMESTAMP PRIMARY KEY NOT NULL,Project VARCHAR(50) NOT NULL, Name VARCHAR(50) NOT NULL, Comment VARCHAR(1024),UNIQUE (Project, Name),CONSTRAINT V6 FOREIGN KEY (Project) REFERENCES Project(Name) ON DELETE CASCADE);\" > /dev/null");
    D_Print("Creating table Image...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Image (Id SERIAL PRIMARY KEY NOT NULL,Name VARCHAR(50) NOT NULL,Width Uint NOT NULL, Height Uint NOT NULL, Channels Cint NOT NULL, Data Bytea NOT NULL,Dix TIMESTAMP NOT NULL,Path VARCHAR(256) NOT NULL,CONSTRAINT V8 FOREIGN KEY (Dix) REFERENCES Dix(Instant) ON DELETE CASCADE);\" > /dev/null");

    D_Print("Altering table Project...\n");
    system("psql -d edix -U edix -c \"ALTER TABLE Project ADD CONSTRAINT V1 CHECK (CDate <= MDate),ADD CONSTRAINT V2 UNIQUE (Settings),ADD CONSTRAINT V3 FOREIGN KEY (Settings) REFERENCES Settings(Id) ON DELETE CASCADE INITIALLY DEFERRED;\" > /dev/null");
    D_Print("Altering table Settings...\n");
    system("psql -d edix -U edix -c \"ALTER TABLE Settings ADD CONSTRAINT V4 UNIQUE (Project),ADD CONSTRAINT V5 FOREIGN KEY (Project) REFERENCES Project(Name) ON DELETE CASCADE INITIALLY DEFERRED;\" > /dev/null");

    D_Print("Creating function CheckTimeFunction...\n");
    system(R"(psql -d edix -U edix -c "CREATE FUNCTION CheckTimeFunction() RETURNS TRIGGER AS \$\$ BEGIN IF NEW.Instant < (SELECT CDate FROM Project p WHERE p.name = NEW.Project) THEN RAISE EXCEPTION 'Cannot insert a dix with Instant < CDate'; END IF; RETURN NEW; END; \$\$ LANGUAGE plpgsql;" > /dev/null)");

    D_Print("Creating trigger on Dix...\n");
    system("psql -d edix -U edix -c \"CREATE TRIGGER CheckTime BEFORE INSERT ON Dix FOR EACH ROW EXECUTE FUNCTION CheckTimeFunction();\" > /dev/null");

    return 0;
}
int checkPostgresService()
{
    PGconn *conn = PQconnectdb("dbname=postgres user=postgres password=");

    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Postgres is not running\n");
        return 1;
    }

    return 0;
}


int addProject(char *name, char *path, char *tup, int tts, char *tpp, bool backup)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione a postgres: %s\n", PQerrorMessage(conn));
        return 1;
    }

    char *query = (char *) malloc(1024 * sizeof(char));
    sprintf(query, "BEGIN;\n"
                   "INSERT INTO Settings (TUP, TTS, TPP, Backup, Project) VALUES ('%s', '%d', '%s', '%s', '%s') RETURNING Id;\n"
                   "INSERT INTO Project (Name, CDate, MDate, Path, Settings) VALUES ('%s', NOW(), NOW(), '%s', (SELECT Id FROM Settings ORDER BY Id DESC LIMIT 1));\n"
                   "COMMIT;\n", tup, tts, tpp, backup ? "TRUE" : "FALSE", name, name, path);

    D_Print("Adding project to Postgres...\n");
    PGresult *res = PQexec(conn, query);
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        const char *sqlstate = PQresultErrorField(res, PG_DIAG_SQLSTATE);

        if (strcmp(sqlstate, "23505") == 0)
            E_Print(RED "POSTGRES Error:" RESET " Questo progetto già esiste!\n");
        else
            E_Print("Errore nell'esecuzione della query: %s\n", PQerrorMessage(conn));

        free(query);
        PQclear(res);
        PQfinish(conn);
        return 1;
    }

    free(query);
    PQclear(res);
    PQfinish(conn);
    return 0;
}
int addDix(char *dixName, char *comment, char **images, char **paths)
{
    char *projectName = getStrFromKey("pName");
    char *projectPath = getStrFromKey("pPath");

    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        free(projectName);
        free(projectPath);
        PQfinish(conn);
        E_Print("Errore di connessione %s\n", PQerrorMessage(conn));
        return 1;
    }

    size_t qSize = 1024;
    char *query = (char *) malloc(qSize * sizeof(char *));
    sprintf(query, "BEGIN;\nINSERT INTO Dix (Instant, Name, Comment, Project) VALUES (NOW(), '%s', '%s', '%s');\n", dixName, comment, projectName);

    char *queryImages;
    if (createQueryImages(projectPath, dixName, images, paths, &queryImages))
    {
        free(projectName);
        free(projectPath);
        free(query);
        PQfinish(conn);
        return 1;
    }

    qSize += strlen(queryImages);
    char *tmp = (char *) realloc(query, qSize * sizeof(char));
    if (tmp == nullptr)
    {
        free(projectName);
        free(projectPath);
        free(query);
        PQfinish(conn);
        E_Print("Errore durante la reallocazione!\n");
        return 1;
    }
    query = tmp;
    sprintf(query + strlen(query), "%sCOMMIT;", queryImages);

    D_Print("Adding dix to project %s...\n", projectName);
    PGresult *res = PQexec(conn, query);
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        const char *sqlstate = PQresultErrorField(res, PG_DIAG_SQLSTATE);

        if (strcmp(sqlstate, "23505") == 0)
            E_Print(" Questo dix già esiste!\n");
        else
            E_Print("Errore nell'esecuzione della query: %s\n", PQerrorMessage(conn));

        FILE *file = fopen("log.txt", "w");
        fwrite(query, strlen(query), 1, file);
        fclose(file);

        free(projectName);
        free(projectPath);
        free(query);
        PQclear(res);
        PQfinish(conn);
        return 1;
    }

    free(query);
    free(projectName);
    free(projectPath);

    return 0;
}
int updateSettings(char *projectName, char *tup, int tts, char *tpp, bool backup)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione a postgres: %s\n", PQerrorMessage(conn));
        return 1;
    }

    char backupChar[2];
    sprintf(backupChar, "%c", backup ? 't' : 'f');

    char query[256];
    sprintf(query, "UPDATE Settings s SET tup = '%s', tts = %d, tpp = '%s' , backup = '%s' WHERE s.Project = '%s'", tup, tts, tpp, backupChar, projectName);

    D_Print("Updating settings...\n");
    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_COMMAND_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Query di UPDATE fallita: %s", PQresultErrorMessage(result));
        return 1;
    }

    PQclear(result);
    PQfinish(conn);
    return 0;
}


int loadProjectOnRedis(char *projectName)
{
    char *tup;
    int tts;
    char *tpp;
    bool backup;
    if (getSettings(projectName, &tup, &tts, &tpp, &backup))
        return 1;

    char *cdate;
    char *mdate;
    char *path;
    if (getProject(projectName, &cdate, &mdate, &path))
        return 1;

    settingsToRedis(tup, tts, tpp, backup);
    projectToRedis(projectName, cdate, mdate, path);


    free(tup);
    free(tpp);
    free(cdate);
    free(mdate);
    free(path);

    return 0;
}
int loadDix(char *dixName, char *projectName)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("(Errore di connessione a postgres), %s\n", PQerrorMessage(conn));
        return 1;
    }

    char query[256];
    sprintf(query, "SELECT Instant FROM Dix d WHERE d.Project = '%s' AND d.Name = '%s';", projectName, dixName);

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return 1;
    }

    int numRows = PQntuples(result);
    if (numRows == 0)
    {
        PQfinish(conn);
        E_Print("Il dix %s non esiste\n", dixName);
        return 1;
    }
    char *instant = strdup(PQgetvalue(result, 0, 0));
    PQclear(result);

    sprintf(query, "SELECT Path, Name, Data, Width, Height, Channels FROM Image i WHERE i.Dix = '%s' ORDER BY Id", instant);
    result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        free(instant);
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return 1;
    }

    numRows = PQntuples(result);

    char *path;
    char *name;
    char *data;
    uint width;
    uint height;
    uint channels;

    for (int i = 0; i < numRows; ++i)
    {
        path = PQgetvalue(result, i, 0);
        name = PQgetvalue(result, i, 1);
        data = PQgetvalue(result, i, 2);
        width = strtoul(PQgetvalue(result, i, 3), nullptr, 10);
        height = strtoul(PQgetvalue(result, i, 4), nullptr, 10);
        channels = strtoul(PQgetvalue(result, i, 5), nullptr, 10);

        char *truePath = (char *) malloc((strlen(path) + strlen(name) + 2) * sizeof(char));
        sprintf(truePath, "%s/%s", path, name);
        buildPath(path);
        unsigned char *img = toImg(data + 2, width, height, channels);
        writeImage(truePath, img, width, height, channels);
    }


    free(instant);
    PQclear(result);
    PQfinish(conn);
    return 0;
}


int delProject(char *name)
{
    char *path;
    if (getProjectPath(name, &path))
        return 1;

    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione %s\n", PQerrorMessage(conn));
        return 1;
    }


    char query[256];
    sprintf(query, "DELETE FROM Project p WHERE p.name = '%s';", name);

    D_Print("Removing project from Postgres...\n");
    PGresult *res = PQexec(conn, query);
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        PQclear(res);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query -> %s", PQerrorMessage(conn));
        return 1;
    }

    PQclear(res);
    PQfinish(conn);

    char command[256];
    sprintf(command, "rm -rf %s", path);
    D_Print("Removing project folder...\n");
    if (system(command) != 0)
    {
        E_Print("Impossibile rimuovere cartella del progetto\n");
        return 1;
    }

    return 0;
}


char *listDixs(char *projectName)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione: %s\n", PQerrorMessage(conn));
        return nullptr;
    }

    char query[256];
    sprintf(query, "SELECT Name, Instant, Comment FROM Dix d WHERE d.Project = '%s' ORDER BY Instant", projectName);
    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return nullptr;
    }

    int numRows = PQntuples(result);

    char *name;
    char *instant;
    char *comment;

    size_t oSize = 256;
    size_t oLen = 5;
    char *out = (char *) malloc(oSize * sizeof(char));
    sprintf(out, "DIX:\n");

    size_t lSize;
    char *line;

    for (int i = 0; i < numRows; ++i)
    {
        name = strdup(PQgetvalue(result, i, 0));
        instant = strdup(PQgetvalue(result, i, 1));
        comment = strdup(PQgetvalue(result, i, 2));
        lSize = strlen(name) + strlen(instant) + strlen(comment) + 256;

        line = (char *) malloc(lSize * sizeof(char));
        sprintf(line, "%s\t%s\t", name, instant);

        uint oldSize = strlen(comment);
        adjustComment(&comment, line, strlen(name) + strlen(instant));

        if (strlen(comment) > oldSize + 256)
        {
            lSize = strlen(name) + strlen(instant) + strlen(comment);
            char *tmp = (char *) realloc(line, lSize * sizeof(char));
            if (tmp == nullptr)
            {
                PQclear(result);
                PQfinish(conn);
                free(out);
                free(line);
                free(name);
                free(instant);
                free(comment);
                E_Print("Errore durante la reallocazione!\n");
                return nullptr;
            }
            line = tmp;
        }
        sprintf(line + strlen(line), "%s\n", comment);

        if (oLen + lSize >= oSize)
        {
            oSize += lSize;
            char *tmp = (char *) realloc(out, oSize * sizeof(char));
            if (tmp == nullptr)
            {
                PQclear(result);
                PQfinish(conn);
                free(out);
                free(line);
                free(name);
                free(instant);
                free(comment);
                E_Print("Errore durante la reallocazione!\n");
                return nullptr;
            }
            out = tmp;
        }
        sprintf(out + oLen, "%s", line);
        oLen += strlen(line);

        free(line);
        free(name);
        free(instant);
        free(comment);
    }

    PQclear(result);
    PQfinish(conn);
    return out;
}
char *listProjects()
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione: %s\n", PQerrorMessage(conn));
        return nullptr;
    }

    char query[256];
    sprintf(query, "SELECT Name, CDate, MDate FROM Project");

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return nullptr;
    }

    int numRows = PQntuples(result);

    size_t oSize = 256;
    size_t oLen = 10;
    char *out = (char *) malloc(oSize * sizeof(char));
    sprintf(out, "PROGETTI:\n");

    char *name;
    char *cdate;
    char *mdate;

    size_t lSize;
    char *line;


    for (int i = 0; i < numRows; ++i)
    {
        name = strdup(PQgetvalue(result, i, 0));
        cdate = strdup(PQgetvalue(result, i, 1));
        mdate = strdup(PQgetvalue(result, i, 2));

        lSize = strlen(name) + strlen(cdate) + strlen(mdate) + 3;
        line = (char *) malloc(lSize * sizeof(char));
        sprintf(line, "%s\t%s\t%s", name, cdate, mdate);

        if (oLen + lSize >= oSize)
        {
            oSize += lSize;
            char *tmp = (char *) realloc(out, oSize * sizeof(char));
            if (tmp == nullptr)
            {
                PQclear(result);
                PQfinish(conn);
                free(out);
                free(line);
                free(name);
                free(cdate);
                free(mdate);
                E_Print("Errore durante la reallocazione!\n");
                return nullptr;
            }
            out = tmp;
        }
        sprintf(out + oLen, "%s\n", line);
        oLen += lSize;

        free(line);
        free(name);
        free(cdate);
        free(mdate);
    }

    PQclear(result);
    PQfinish(conn);

    return out;
}


int getSettings(char *projectName, char **tup, int *tts, char **tpp, bool *backup)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione: %s\n", PQerrorMessage(conn));
        return 1;
    }

    char query[256];
    sprintf(query, "SELECT TUP, TTS, TPP, Backup FROM Settings s WHERE s.Project = '%s'", projectName);

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return 1;
    }

    *tup = strdup(PQgetvalue(result, 0, 0));
    *tts = (int) strtol(PQgetvalue(result, 0, 1), nullptr, 10);
    *tpp = strdup(PQgetvalue(result, 0, 2));
    *backup = strcmp(PQgetvalue(result, 0, 3), "t") == 0;

    PQclear(result);
    PQfinish(conn);
    return 0;
}
int getProject(char *projectName, char **cdate, char **mdate, char **path)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione: %s\n", PQerrorMessage(conn));
        return 1;
    }

    char query[256];
    sprintf(query, "SELECT CDate, Mdate, Path FROM Project p WHERE p.Name = '%s'", projectName);

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return 1;
    }

    *cdate = strdup(PQgetvalue(result, 0, 0));
    *mdate = strdup(PQgetvalue(result, 0, 1));
    *path = strdup(PQgetvalue(result, 0, 2));

    PQclear(result);
    PQfinish(conn);
    return 0;
}
int getProjectPath(char *projectName, char **path)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione %s\n", PQerrorMessage(conn));
        return 1;
    }

    char query[256];
    sprintf(query, "SELECT path FROM Project p WHERE p.name = '%s'", projectName);

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return 1;
    }

    *path = strdup(PQgetvalue(result, 0, 0));

    PQclear(result);

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
bool existProject(char *name)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione: %s\n", PQerrorMessage(conn));
        return false;
    }

    char query[256];
    sprintf(query, "SELECT Name FROM Project p Where p.Name = '%s' ", name);

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return false;
    }


    int numRows = PQntuples(result);


    PQclear(result);
    PQfinish(conn);
    return numRows == 1;
}


int createQueryImages(char *projectPath, char *dixName, char **names, char **paths, char **query)
{
    size_t qSize = 128;
    size_t qLen = 0;
    *query = (char *) malloc(qSize * sizeof(char));

    for (int i = 0; paths[i] != nullptr; ++i)
    {
        char *path = (char *) malloc((strlen(projectPath) + strlen(dixName) + strlen(names[i]) + 8) * sizeof(char));
        sprintf(path, "%s/.dix/%s/%s", projectPath, dixName, names[i]);


        uint width;
        uint height;
        uint channels;
        unsigned char *image = loadImage(path, &width, &height, &channels);
        char *imageData = toExadecimal(image, width, height, channels);

        size_t lSize = strlen(imageData) + strlen(names[i]) + strlen(paths[i]) + 113;
        char *line = (char *) malloc(lSize * sizeof(char));
        sprintf(line, "INSERT INTO Image (Name, Dix, Path, Data, Width, Height, Channels) VALUES ('%s', NOW(), '%s', E'\\\\x%s', %d, %d, %d);\n", names[i], paths[i], imageData, width, height, channels);

        qSize += lSize;
        char *tmp = (char *) realloc(*query, qSize * sizeof(char));
        if (tmp == nullptr)
        {
            free(path);
            free(image);
            free(imageData);
            free(line);
            free(*query);
            E_Print("Errore nella reallocazione!\n");
            return 1;
        }
        *query = tmp;
        sprintf(*query + qLen, "%s", line);
        qLen += lSize;
        free(path);
        free(image);
        free(imageData);
        free(line);
    }

    return 0;
}
char *toExadecimal(unsigned char *imageData, uint width, uint height, uint channels)
{
    uint oSize = width * height * channels * 2;
    char *out = (char *) malloc(oSize * sizeof(char));

#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static) default(none) shared(imageData, oSize, out)
    for (size_t i = 0; i < oSize / 2; i++)
        sprintf(out + (i * 2), "%02X", imageData[i]);

    return out;
}
unsigned char *toImg(char *exaData, uint width, uint height, uint channels)
{
    auto *out = (unsigned char *) malloc(width * height * channels * sizeof(unsigned char));
    size_t eLen = strlen(exaData);

#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static) default(none) shared(eLen, exaData, out)
    for (int i = 0; i < eLen; i += 2)
    {
        char byte[3] = {exaData[i], exaData[i + 1], '\0'};
        out[i / 2] = strtoul(byte, nullptr, 16);
    }

    return out;
}
int adjustComment(char **comment, char *line, uint padding)
{
    char *out = (char *) malloc((strlen(*comment) + strlen(line)) * sizeof(char));
    if (out == nullptr)
    {
        E_Print("Errore durante l'allocazione!\n");
        return 1;
    }
    char *token = strtok(*comment, "\n");

    if (token == nullptr)
        return 0;

    sprintf(out, "%s\n", token);
    for (token = strtok(nullptr, "\n"); token != nullptr; token = strtok(nullptr, "\n"))
    {
        for (int i = 0; i < padding; ++i)
            if (line[i] == '\t')
                strcat(out, "\t");
            else
                strcat(out, " ");
        sprintf(out + strlen(out), "\t%s\n", token);
    }
    free(*comment);
    *comment = out;

    return 0;
}
int buildPath(char *path)
{
    char *projectPath = getStrFromKey("pPath");
    if (strcmp(path, projectPath) == 0)
        return 0;

    char command[256];
    sprintf(command, "mkdir -p %s", path);
    if (system(command) != 0)
    {
        free(projectPath);
        E_Print("Errore durante la creazione della cartella %s\n", path);
        return 1;
    }


    free(projectPath);
    return 0;
}
