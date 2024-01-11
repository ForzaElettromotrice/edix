#include "temp.h"



int set_check(char *name, redisReply *reply, redisContext *context){
    if (reply == nullptr) {
        printf("Errore nell'esecuzione del comando SET %s\n",name);
        freeReplyObject(reply);
        redisFree(context);
        return 1;
    }

    freeReplyObject(reply);
    return 0;
}

int get_check(char *name, redisReply * reply, redisContext *context){
    if (reply == NULL) {
        printf("Errore nell'esecuzione del comando GET %s\n",name);
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

int upload_to_redis(int id, int project, Modex_t mod_ex, u_int tts, bool vcs, Compx_t comp, Tppx_t tpp, Tup_t tup) {
    // Connetti a Redis
    redisContext *context = redisConnect("localhost", 6379);

    if (context == NULL || context->err) {
        if (context) {
            printf("Errore di connessione a Redis: %s\n", context->errstr);
            redisFree(context);
            return 1;
        } else {
            printf("Impossibile inizializzare la connessione a Redis\n");
            return 1;
        }
    }

    printf("Connesso a Redis\n");

    //Esegui i comandi per caricare le settings su redis
    redisReply *reply = (redisReply*)redisCommand(context, "SET %s %d","ID",id);
    set_check("ID",reply, context);

    reply = (redisReply*)redisCommand(context, "SET %s %d","Project",project);
    set_check("Project",reply, context);

    reply = (redisReply*)redisCommand(context, "SET %s %s","Mod_ex",mod_ex);
    set_check("Mod_ex",reply, context);

    reply = (redisReply*)redisCommand(context, "SET %s %u","TTS",tts);
    set_check("TTS",reply, context);

    reply = (redisReply*)redisCommand(context, "SET %s %d","VCS",vcs);
    set_check("VCS",reply, context);

    reply = (redisReply*)redisCommand(context, "SET %s %s","COMP",comp);
    set_check("COMP",reply, context);

    reply = (redisReply*)redisCommand(context, "SET %s %s","TPP",tpp);
    set_check("TPP",reply, context);

    reply = (redisReply*)redisCommand(context, "SET %s %s","TUP",tup);
    set_check("TUP",reply, context);


    printf("Dati salvati correttamente\n");

    // Recupera i dati dalle chiavi
    reply = (redisReply*)redisCommand(context, "GET %s", "ID");
    get_check("ID",reply, context);

    reply = (redisReply*)redisCommand(context, "GET %s", "Project");
    get_check("Project",reply, context);

    reply = (redisReply*)redisCommand(context, "GET %s", "Mod_ex");
    get_check("Mod_ex",reply, context);

    reply = (redisReply*)redisCommand(context, "GET %s", "TTS");
    get_check("TTS",reply, context);

    reply = (redisReply*)redisCommand(context, "GET %s", "VCS");
    get_check("VCS",reply, context);

    reply = (redisReply*)redisCommand(context, "GET %s", "COMP");
    get_check("COMP",reply, context);

    reply = (redisReply*)redisCommand(context, "GET %s", "TPP");
    get_check("TPP",reply, context);

    reply = (redisReply*)redisCommand(context, "GET %s", "TUP");
    get_check("TUP",reply, context);

    return 0;

}

int get_id(char *projectName, char **ID){
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");

    // Verifica lo stato della connessione
    if (PQstatus(conn) != CONNECTION_OK) {
        fprintf(stderr, "Errore di connessione: %s\n", PQerrorMessage(conn));
        PQfinish(conn);
        return 1;
    }

    // Esegui una query SQL
    char query[256];
    sprintf(query, "SELECT ID FROM Project WHERE Name = '%s'", projectName);

    PGresult *result = PQexec(conn, query);

    // Verifica lo stato della query
    if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        PQclear(result);
        PQfinish(conn);
        return 1;
    }

    // Recupera e stampa i risultati
    int numRows = PQntuples(result);
    int numCols = PQnfields(result);

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            *ID = PQgetvalue(result, i, j);
            printf("%s\t", PQgetvalue(result, i, j));
        }
        printf("\n");
    }

    //Libera la memoria delle risorse
    PQclear(result);
    PQfinish(conn);

    return 0;
}

int get_from_settings(char *projectName) {
    // Crea una connessione al database
    PGconn *conn = PQconnectdb("host=localhost dbname=edix user=edix password=");

    // Verifica lo stato della connessione
    if (PQstatus(conn) != CONNECTION_OK) {
        fprintf(stderr, "Errore di connessione: %s\n", PQerrorMessage(conn));
        PQfinish(conn);
        return 1;
    }

    char *projectId;
    get_id(projectName, &projectId);

    // Esegui una query SQL
    char query[256];
    sprintf(query, "SELECT * FROM Settings_p WHERE Project = %s", projectId);

    printf("%s\n", query);

    PGresult *result = PQexec(conn, query);

    // Verifica lo stato della query
    if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        fprintf(stderr, "Errore nell'esecuzione della query: %s\n", PQresultErrorMessage(result));
        PQclear(result);
        PQfinish(conn);
        return 1;
    }

    // Recupera e stampa i risultati
    int numRows = PQntuples(result);
    int numCols = PQnfields(result);

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%s\t", PQgetvalue(result, i, j));
        }
        printf("\n");
    }

//     Libera la memoria delle risorse
    PQclear(result);
    PQfinish(conn);

    return 0;
}