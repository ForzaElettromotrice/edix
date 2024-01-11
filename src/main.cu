#include "main.h"


int main(int argc, char *argv[])
{
//    setup();
    // TODO: setup di tutte le risorse necessarie (tipo redis, o comunicatore col db)
    // TODO: magari puoi aprire direttamente un progetto passandolo come argomento
    switch (argc)
    {
        case 1:
            inputLoop();
            break;
        default:
            fprintf(stderr, RED "usage:" RESET " ./edix\n");
            exit(EXIT_FAILURE);
    }

    //TODO: fare tutte le cose da fare prima di chiudere

//    get_from_settings("Progetto2");

    return 0;
}
int inputLoop()
{
    size_t lineSize = 256;
    char *line = (char *) malloc(256);

    size_t bytesRead;
    Env env = HOMEPAGE;
    bool stop = false;

    while (!stop && ((int)(bytesRead = getline(&line, &lineSize, stdin))) != -1 )
    {
        line[bytesRead - 1] = '\0';
        switch (env)
        {
            case HOMEPAGE:
                parseHome(line, &env);
                break;
            case PROJECT:
                parseProj(line, &env);
                break;
            case SETTINGS:
                break;
            case EXIT:
                //Unreachable
                break;
        }
        if(env == EXIT)
            stop = true;
    }

    free(line);

    return 0;
}


int setup()
{
    //TODO: mettere i controlli per vedere se non esiste già
    D_PRINT("Creating user edix...");
    system("createuser edix -d > /dev/null");
    D_PRINT("Creating database...");
    system("createdb edix -O edix > /dev/null");

    D_PRINT("Creating Tupx_t...");
    system("psql -d edix -U edix -c \"CREATE TYPE Tupx_t AS ENUM ('Bilinear', 'Bicubic');\" > /dev/null");
    D_PRINT("Creating Modex_t...");
    system("psql -d edix -U edix -c \"CREATE TYPE Modex_t AS ENUM('Immediate', 'Programmed');\" > /dev/null");
    D_PRINT("Creating Compx_t...");
    system("psql -d edix -U edix -c \"CREATE TYPE Compx_t AS ENUM('JPEG', 'PNG', 'PPM');\" > /dev/null");
    D_PRINT("Creating Tppx_t...");
    system("psql -d edix -U edix -c \"CREATE TYPE Tppx_t AS ENUM('Serial', 'OMP', 'CUDA');\" > /dev/null");

    D_PRINT("Creating Tupx...");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Tupx AS Tupx_t;\" > /dev/null");
    D_PRINT("Creating Modex...");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Modex AS Modex_t;\" > /dev/null");
    D_PRINT("Creating Compx...");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Compx AS Compx_t;\" > /dev/null");
    D_PRINT("Creating Tppx...");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Tppx AS Tppx_t;\" > /dev/null");
    D_PRINT("Creating Uint...");
    system("psql -d edix -U edix -c \"CREATE DOMAIN Uint AS integer CHECK(VALUE >= 0);\" > /dev/null");

    D_PRINT("Creating table Project...");
    system("psql -d edix -U edix -c \"CREATE TABLE Project (Id SERIAL PRIMARY KEY NOT NULL,Name VARCHAR(50) NOT NULL,CDate TIMESTAMP NOT NULL,MDate TIMESTAMP NOT NULL,Path VARCHAR(256) UNIQUE NOT NULL,Settings INT NOT NULL);\" > /dev/null");
    D_PRINT("Creating table Settings_p...");
    system("psql -d edix -U edix -c \"CREATE TABLE Settings_p (Id SERIAL PRIMARY KEY NOT NULL,TUP Tupx NOT NULL,Mod_ex Modex NOT NULL,Comp Compx NOT NULL,TTS INT NOT NULL,TPP Tppx NOT NULL,VCS BOOLEAN NOT NULL,Project INT NOT NULL);\" > /dev/null");

    D_PRINT("Altering table Project...");
    system("psql -d edix -U edix -c \"ALTER TABLE Project ADD CONSTRAINT V1 CHECK (CDate <= MDate),ADD CONSTRAINT V2 UNIQUE (Settings),ADD CONSTRAINT V3 FOREIGN KEY (Settings) REFERENCES Settings_p(Id) INITIALLY DEFERRED;\" > /dev/null");
    D_PRINT("Altering table Settings_p...");
    system("psql -d edix -U edix -c \"ALTER TABLE Settings_p ADD CONSTRAINT V4 UNIQUE (Project),ADD CONSTRAINT V5 FOREIGN KEY (Project) REFERENCES Project(Id) INITIALLY DEFERRED;\" > /dev/null");

    D_PRINT("Creating table Dix...");
    system("psql -d edix -U edix -c \"CREATE TABLE Dix (Instant TIMESTAMP PRIMARY KEY NOT NULL,Project INT NOT NULL,CONSTRAINT V6 FOREIGN KEY (Project) REFERENCES Project(Id));\" > /dev/null");
    D_PRINT("Creating table Photo...");
    system("psql -d edix -U edix -c \"CREATE TABLE Photo (Id SERIAL PRIMARY KEY NOT NULL,Name VARCHAR(50) NOT NULL,Path VARCHAR(256) NOT NULL,Comp Compx NOT NULL,Project INT,Dix TIMESTAMP,CONSTRAINT V7 FOREIGN KEY (Project) REFERENCES Project(Id),CONSTRAINT V8 FOREIGN KEY (Dix) REFERENCES Dix(Instant),CONSTRAINT V9 CHECK ((Project IS NOT NULL AND Dix IS NULL) OR (Project IS NULL AND Dix IS NOT NULL)));\" > /dev/null");
}