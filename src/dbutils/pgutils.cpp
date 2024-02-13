//
// Created by f3m on 12/01/24.
//
#include "pgutils.hpp"


void hexStringToByteArray(const char *hexString, unsigned char **byteArray, size_t *length)
{
    size_t strLength = strlen(hexString);
    *length = strLength / 2; // Ogni coppia di caratteri esadecimali corrisponde a un byte
    *byteArray = (unsigned char *) malloc(*length);
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static) default(none) shared(length, hexString, byteArray)
    for (size_t i = 0; i < *length; i++)
        sscanf(hexString + 2 * i, "%2hhx", &(*byteArray)[i]);
}

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

    D_Print("Creating table Project...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Project (Name VARCHAR(50) PRIMARY KEY NOT NULL,CDate TIMESTAMP NOT NULL,MDate TIMESTAMP NOT NULL,Path VARCHAR(256) UNIQUE NOT NULL,Settings INT NOT NULL);\" > /dev/null");
    D_Print("Creating table Settings...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Settings (Id SERIAL PRIMARY KEY NOT NULL,TUP Tupx NOT NULL,TTS Uint5 NOT NULL,TPP Tppx NOT NULL,Backup BOOLEAN NOT NULL,Project VARCHAR(50) NOT NULL);\" > /dev/null");
    D_Print("Creating table Dix...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Dix (Instant TIMESTAMP PRIMARY KEY NOT NULL,Project VARCHAR(50) NOT NULL, Name VARCHAR(50) NOT NULL, Comment VARCHAR(1024),UNIQUE (Project, Name),CONSTRAINT V6 FOREIGN KEY (Project) REFERENCES Project(Name) ON DELETE CASCADE);\" > /dev/null");
    D_Print("Creating table Image...\n");
    system("psql -d edix -U edix -c \"CREATE TABLE Image (Id SERIAL PRIMARY KEY NOT NULL,Name VARCHAR(50) NOT NULL,Data Bytea NOT NULL,Dix TIMESTAMP NOT NULL,Path VARCHAR(256) NOT NULL,CONSTRAINT V8 FOREIGN KEY (Dix) REFERENCES Dix(Instant) ON DELETE CASCADE);\" > /dev/null");

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

int loadProjectOnRedis(char *projectName)
{

    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione a postgres), %s\n", PQerrorMessage(conn));
        return 1;
    }


    char **settings = getSettings(conn, projectName);
    if (settings == nullptr)
        return 1;

    char **project = getProject(conn, projectName);
    if (project == nullptr)
        return 1;

    projectToRedis(project[0],
                   project[1],
                   project[2],
                   project[3],
                   (int) strtol(project[4], nullptr, 10));

    settingsToRedis(
            settings[1],
            strtoul(settings[2], nullptr, 10),
            settings[3],
            strcmp("t", settings[4]) == 0);


    for (int i = 0; settings[i] != nullptr; ++i)
        free(settings[i]);
    free(settings);

    for (int i = 0; project[i] != nullptr; ++i)
        free(project[i]);
    free(project);

    PQfinish(conn);

    return 0;
}
int loadDix(char *name, char *projectName, char *pPath)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("(Errore di connessione a postgres), %s\n", PQerrorMessage(conn));
        return 1;
    }

    char query[256];
    sprintf(query, "SELECT Instant FROM Dix d WHERE d.Project = '%s' AND d.Name = '%s';", projectName, name);

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
        E_Print("Il dix %s non esiste\n", name);
        return 1;
    }
    char *instant = PQgetvalue(result, 0, 0);

    PQclear(result);

    sprintf(query, "SELECT Path, Name, Data FROM Image i WHERE i.Dix = '%s' ORDER BY Id", instant);

    result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return 1;
    }

    numRows = PQntuples(result);

    char *image;
    char *path;
    char *iName;

    for (int i = 0; i < numRows; ++i)
    {
        path = PQgetvalue(result, i, 0);
        iName = PQgetvalue(result, i, 1);
        image = PQgetvalue(result, i, 2);

        char *truePath = (char *) malloc((strlen(iName) + strlen(path)) * sizeof(char));
        sprintf(truePath, "%s/%s", path, iName);
        checkPath(path, pPath);
        saveImage(truePath, image);

        free(truePath);
    }


    PQfinish(conn);

    return 0;
}
int addProject(char *name, char *path, char *TPP, char *TUP, uint TTS, bool Backup)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione a postgres: %s\n", PQerrorMessage(conn));
        return 1;
    }

    char query[1024];
    sprintf(query, "BEGIN;\n"
                   "INSERT INTO Settings (TUP, TTS, TPP, Backup, Project) VALUES ('%s', '%d', '%s', '%s', '%s') RETURNING Id;\n"
                   "INSERT INTO Project (Name, CDate, MDate, Path, Settings) VALUES ('%s', NOW(), NOW(), '%s', (SELECT Id FROM Settings ORDER BY Id DESC LIMIT 1));\n"
                   "COMMIT;\n", TUP, TTS, TPP, Backup ? "TRUE" : "FALSE", name, name, path);
    D_Print("Adding project to Postgres...\n");
    PGresult *res = PQexec(conn, query);

    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        const char *sqlstate = PQresultErrorField(res, PG_DIAG_SQLSTATE);

        if (strcmp(sqlstate, "23505") == 0)
            E_Print(RED "POSTGRES Error:" RESET " Questo progetto già esiste!\n");
        else
            E_Print("Errore nell'esecuzione della query: %s\n", PQerrorMessage(conn));


        PQclear(res);
        PQfinish(conn);
        return 1;
    }


    PQclear(res);
    PQfinish(conn);
    return 0;
}
int addDix(char *projectName, char *dixName, char *comment, char **images, char **paths)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione %s\n", PQerrorMessage(conn));
        return 1;
    }

    char *pPath = getStrFromKey("pPath");


    size_t qSize = 1024;
    char *query = (char *) malloc(1024 * sizeof(char *));
    sprintf(query, "BEGIN;\nINSERT INTO Dix (Instant, Name, Comment, Project) VALUES (NOW(), '%s', '%s', '%s');\n",
            dixName,
            comment, projectName);


    for (int i = 0; paths[i] != nullptr; ++i)
    {
        D_Print("path = %s\timage = %s\n", paths[i], images[i]);
        size_t iSize;
        char truePath[256];
        sprintf(truePath, "%s/.dix/%s/%s", pPath, dixName, images[i]);

        unsigned char *imageData = getImageData(truePath, &iSize);
        if (imageData == nullptr)
        {
            PQfinish(conn);
            continue;
        }

        char *line = (char *) malloc((iSize * 2 + 256) * sizeof(char));
        sprintf(line, "INSERT INTO Image (Name, Dix, Path, Data) VALUES ('%s', NOW(), '%s', E'\\\\x",
                images[i], paths[i]);

        uint len = strlen(line);
        FILE *file = fopen("log.txt", "w");
        fwrite(imageData, iSize, 1, file);
        fclose(file);
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static) default(none) shared(iSize, line, imageData, len)
        for (int j = 0; j < iSize; j++)
            sprintf(line + len + j * 2, "%02x", imageData[j]);
        file = fopen("log2.txt", "w");
        fwrite(line, strlen(line), 1, file);
        fclose(file);

        strcat(line, "');");


        qSize += iSize * 2 + 256;
        char *tmp = (char *) realloc(query, qSize * sizeof(char));
        if (tmp == nullptr)
        {
            PQfinish(conn);
            free(imageData);
            free(query);
            free(line);
            free(pPath);
            E_Print("Error while reallocating!\n");
            return 1;
        }
        query = tmp;

        strcat(query, line);

        free(imageData);
        free(line);
        D_Print("Fine for\n");
    }

    free(pPath);


    qSize += 256;
    char *tmp = (char *) realloc(query, qSize * sizeof(char));
    if (!tmp)
    {
        free(query);
        PQfinish(conn);
        E_Print("Error while reallocating!\n");
        return 1;
    }
    query = tmp;

    strcat(query, "COMMIT;");

    D_Print("Adding dix to project %s...\n", projectName);
    PGresult *res = PQexec(conn, query);

    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        const char *sqlstate = PQresultErrorField(res, PG_DIAG_SQLSTATE);

        if (strcmp(sqlstate, "23505") == 0)
            E_Print(RED "Error:" RESET " Questo dix già esiste!\n");
        else
            E_Print("Errore nell'esecuzione della query: %s\n", PQerrorMessage(conn));

        FILE *file = fopen("log.txt", "w");
        fwrite(query, strlen(query), 1, file);
        fclose(file);
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
int delProject(char *name)
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione %s\n", PQerrorMessage(conn));
        return 1;
    }

    char *path = getPath(conn, name);
    if (path == nullptr)
    {
        PQfinish(conn);
        return 1;
    }

    D_Print("Removing project folder...\n");
    char command[256];
    sprintf(command, "rm -rf %s", path);
    if (system(command) != 0)
    {
        PQfinish(conn);
        E_Print("Impossibile rimuovere cartella del progetto\n");
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

    return 0;
}
int updateSettings(int id, char *tup, u_int tts, char *tpp, bool backup, char *pName)
{

    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");
    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione a postgres: %s\n", PQerrorMessage(conn));
        return 1;
    }
    char backupChar[256];

    if (backup == 0)
    {
        sprintf(backupChar, "f");
    } else
    {
        sprintf(backupChar, "t");
    }


    char query[256];

    sprintf(query,
            "UPDATE Settings SET id = %d, tup = '%s', tts = %u, tpp = '%s' , backup = '%s' , project = '%s' WHERE Id = %d",
            id,
            tup,
            tts,
            tpp,
            backupChar,
            pName,
            id);

    PGresult *result = PQexec(conn, query);

    if (PQresultStatus(result) != PGRES_COMMAND_OK)
    {
        E_Print("Query di UPDATE fallita: %s", PQresultErrorMessage(result));
        PQclear(result);
        PQfinish(conn);
        return 1;
    }
    D_Print("query update eseguita su settings\n");
    PQclear(result);
    PQfinish(conn);

    return 0;
}


char *getProjects()
{
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");


    if (PQstatus(conn) != CONNECTION_OK)
    {
        PQfinish(conn);
        E_Print("Errore di connessione: %s\n", PQerrorMessage(conn));
        return nullptr;
    }


    char query[256];
    sprintf(query, "SELECT Name FROM Project");

    PGresult *result = PQexec(conn, query);
    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        PQclear(result);
        PQfinish(conn);
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
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
    //TODO: da testare
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
    int numCols = PQnfields(result);

    //TODO: formattare per bene la stringa finale

    char *names = (char *) malloc(1024 * sizeof(char));
    sprintf(names, BLUE BOLD "Dix su %s:\n" RESET, projectName);

    for (int i = 0; i < numRows; i++)
    {
        strcat(names, YELLOW "- ");
        for (int j = 0; j < numCols; j++)
        {
            char *line = PQgetvalue(result, i, j);
            if (j == numCols - 1)
                changeFormat(&line);
            strcat(names, line);
            strcat(names, "\t");
        }
        strcat(names, "\n" RESET);
    }

    PQclear(result);
    PQfinish(conn);
    return names;
}
char **getProject(PGconn *conn, char *projectName)
{
    char query[256];
    sprintf(query, "SELECT * FROM Project WHERE Name = '%s'", projectName);

    PGresult *result = PQexec(conn, query);


    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
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
char **getSettings(PGconn *conn, char *projectName)
{
    char query[256];
    sprintf(query, "SELECT * FROM Settings WHERE Project = '%s'", projectName);

    PGresult *result = PQexec(conn, query);


    if (PQresultStatus(result) != PGRES_TUPLES_OK)
    {
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
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
        E_Print("Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        return nullptr;
    }

    char *path = strdup(PQgetvalue(result, 0, 0));

    PQclear(result);
    return path;
}
unsigned char *getImageData(char *path, size_t *dim)
{
    FILE *file = fopen(path, "rb");
    if (!file)
    {
        E_Print(RED "POSTGRES Error: " RESET "Errore nell'apertura del file -> %s\n", strerror(errno));
        return nullptr;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    auto *bytea_data = (unsigned char *) malloc(file_size * sizeof(unsigned char));


    if (bytea_data == nullptr)
    {
        fclose(file);
        E_Print(RED "POSTGRES Error:" RESET "Errore nell'allocazione di memoria\n");
        return nullptr;
    }

    fread(bytea_data, 1, file_size, file);
    fclose(file);
    *dim = file_size;

    return bytea_data;
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
void changeFormat(char **comment)
{
    char *out = (char *) malloc((strlen(*comment) + 256) * sizeof(char));
    char *token = strtok(*comment, "\n");

    if (token == nullptr)
        return;

    sprintf(out, "%s", token);
    for (token = strtok(nullptr, "\n"); token != nullptr; token = strtok(nullptr, "\n"))
    {
        strcat(out, "\n\t\t\t\t\t");
        strcat(out, token);
    }

    *comment = out;
}
int checkPath(char *path, char *pPath)
{
    if (strcmp(path, pPath) == 0)
        return 0;

    D_Print("Creating folder %s...\n", path);
    char *command = (char *) malloc((strlen(path) + 10) * sizeof(char));
    sprintf(command, "mkdir -p %s", path);

    if (system(command) != 0)
    {
        free(command);
        E_Print("Error while creating dir!\n");
        return 1;
    }

    return 0;
}
int saveImage(char *path, char *img)
{
    FILE *file = fopen(path, "wb");
    if (!file)
    {
        E_Print(RED "POSTGRES Error: " RESET "Errore nell'apertura del file -> %s\n", strerror(errno));
        return 1;
    }

    unsigned char *byteArray;
    size_t len;
    hexStringToByteArray(img + 2, &byteArray, &len);


    fwrite(byteArray, sizeof(unsigned char), len, file);
    fclose(file);

    free(byteArray);

    return 0;
}
