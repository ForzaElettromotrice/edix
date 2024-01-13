
#include "rdutils.hpp"
/*
    Operation IDs delle frocerie:
    -Upscaling
    -Downscaling
    -Composizione
    -Blur
    -GrayScale
    -Sovrapposizione


*/
//TODO TEST DELLE FUNCS
//initialize connection to redis
int initRedis(redisContext **context)
{
    *context = redisConnect("localhost", 6379);

    if (*context == nullptr || (*context)->err)
    {
        if (*context)
        {
            fprintf(stderr, "Error while connecting to redis: %s\n", (*context)->errstr);
            redisFree(*context);
        } else
            fprintf(stderr, "Unable to initialize connection to redis\n");


        return 1;
    }
    return 0;
}
//check if set redis command is valid
int setChecking(char *name, redisReply *reply, redisContext *context)
{
    if (reply == nullptr)
    {
        freeReplyObject(reply);
        redisFree(context);
        handle_error("Error while executing redis command SET %s\n", name);
    }

    return 0;
}

//check if get redis command is valid
int getChecking(char *name, redisReply *reply, redisContext *context)
{
    if (reply == nullptr)
    {
        printf("Errore nell'esecuzione del comando GET %s\n", name);
        freeReplyObject(reply);
        redisFree(context);
        return 1;
    }
    return 0;
}

//cache all settings of a project to redis
int settingsToRedis(int id, char *tup, char *mod_ex, char *comp, u_int tts, char *tpp, bool vcs, int project)
{
    //Init redis connection
    redisContext *context;
    initRedis(&context);

    //send settings to redis
    auto *reply = (redisReply *) redisCommand(context, "SET ID %d", id);
    setChecking((char *) "ID", reply, context);


    reply = (redisReply *) redisCommand(context, "SET Project %d", project);
    setChecking((char *) "Project", reply, context);


    reply = (redisReply *) redisCommand(context, "SET Mod_ex %s", mod_ex);
    setChecking((char *) "Mod_ex", reply, context);

    reply = (redisReply *) redisCommand(context, "SET TTS %u", tts);
    setChecking((char *) "TTS", reply, context);

    reply = (redisReply *) redisCommand(context, "SET VCS %d", vcs);
    setChecking((char *) "VCS", reply, context);


    reply = (redisReply *) redisCommand(context, "SET COMP %s", comp);
    setChecking((char *) "COMP", reply, context);

    reply = (redisReply *) redisCommand(context, "SET TPP %s", tpp);
    setChecking((char *) "TPP", reply, context);


    reply = (redisReply *) redisCommand(context, "SET TUP %s", tup);
    setChecking((char *) "TUP", reply, context);


    //End redis connection

    freeReplyObject(reply);
    redisFree(context);

    return 0;
}

//retrieve all settings cached in redis
int settingsFromRedis(int *id, char **tup, char **mod_ex, char **comp, u_int *tts, char **tpp, bool *vcs, int *project)
{
    //init redis connection
    redisContext *context;
    initRedis(&context);

    //TODO: controllare i return di getChecking...
    //TODO: poi sta funzione non me piace, ce deve esse un modo piu intelligente di farla

    // Recupera i dati dalle chiavi
    auto *reply = (redisReply *) redisCommand(context, "GET ID");
    getChecking((char *) "ID", reply, context);
    *id = (int) reply->integer;

    reply = (redisReply *) redisCommand(context, "GET Project");
    getChecking((char *) "Project", reply, context);
    *project = (int) reply->integer;

    reply = (redisReply *) redisCommand(context, "GET Mod_ex");
    getChecking((char *) "Mod_ex", reply, context);
    *mod_ex = reply->str;

    reply = (redisReply *) redisCommand(context, "GET TTS");
    getChecking((char *) "TTS", reply, context);
    *tts = reply->integer;

    reply = (redisReply *) redisCommand(context, "GET VCS");
    getChecking((char *) "VCS", reply, context);
    *vcs = reply->integer == 1;

    reply = (redisReply *) redisCommand(context, "GET COMP");
    getChecking((char *) "COMP", reply, context);
    *comp = reply->str;

    reply = (redisReply *) redisCommand(context, "GET TPP");
    getChecking((char *) "TPP", reply, context);
    *tpp = reply->str;

    reply = (redisReply *) redisCommand(context, "GET TUP");
    getChecking((char *) "TUP", reply, context);
    *tup = reply->str;

    //TODO: printa tutti i valori e compara con i SET
    //End redis connection
    freeReplyObject(reply);
    redisFree(context);

    return 0;

}

//TODO GET FROM REDIS USING KEY
redisReply* getFromKey(char *key){
    //initialize connection
    redisContext *context;
    initRedis(&context);

    //todo get from key

    //end connection;
    redisFree(context);

}