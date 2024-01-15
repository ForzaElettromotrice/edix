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
//check if connection is alive
int checkRedisService(){

    redisContext *context;
    context = redisConnect("localhost", 6379);

    if (context == nullptr || (context)->err)
    {
        if (context)
        {
            fprintf(stderr, "Error while connecting to redis: %s\n", (context)->errstr);
            redisFree(context);
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
        handle_error("Error while executing redis SET command %s\n", name);
    }

    return 0;
}

//check if get redis command is valid
int getChecking(char *name, redisReply *reply, redisContext *context)
{
    if (reply == nullptr)
    {
        handle_error("Error while executing GET command %s\n", name);
        freeReplyObject(reply);
        redisFree(context);
        return 1;
    }
    return 0;
}

//cache project infos on redis
int projectToRedis(char *name, char *cDate, char *mDate, char *path, int settings){
    //initialize conn with redis
    redisContext *context;
    initRedis(&context)

    //send project to redis
    setKeyValueStr((char *)"Name",name);
    setKeyValueStr((char *)"CDate",cDate);
    setKeyValueStr((char *)"MDate",mDate);
    setKeyValueStr((char *)"pPath",path);
    setKeyValueInt((char *)"Settings",settings);

    //end redis connection
    redisFree(context);

    return 0;
}

//cache all settings of a project to redis
int settingsToRedis(int id, char *tup, char *mod_ex, char *comp, u_int tts, char *tpp, bool vcs, char  *pName)
{
    //Init redis connection
    redisContext *context;
    initRedis(&context);

    //send settings to redis
    setKeyValueInt((char *)"ID",id);
    setKeyValueStr((char *)"Project",pName);
    setKeyValueStr((char *)"Mod_ex",mod_ex);
    setKeyValueInt((char *)"TTS",tts);
    setKeyValueInt((char *)"VCS",vcs);
    setKeyValueStr((char *)"COMP",comp);
    setKeyValueStr((char *)"TPP",tpp);
    setKeyValueStr((char *)"TUP",tup);

    //End redis connection
    redisFree(context);

    return 0;
}


//retrieve all settings of a project cached in redis
int settingsFromRedis(int *id, char **tup, char **mod_ex, char **comp, u_int *tts, char **tpp, bool *vcs, char **pName)
{
    //init redis connection
    redisContext *context;
    initRedis(&context);

    //TODO: controllare i return di getChecking...
    //TODO: poi sta funzione non me piace, ce deve esse un modo piu intelligente di farla

    // Recupera i dati dalle chiavi
 
    *id = getIntFromKey((char *)"ID");

    *pName = getStrFromKey((char *)"Project");

    *mod_ex = getStrFromKey((char *)"Mod_ex");

    *tts = getIntFromKey((char *)"TTS");

    auto *reply = (redisReply *) redisCommand(context, "GET VCS");
    getChecking((char *) "VCS", reply, context);
    *vcs = reply->integer == 1;

    *comp = getStrFromKey((char *)"COMP");

    *tpp = getStrFromKey((char *)"TPP");

    *tup = getStrFromKey((char *)"TUP");

    //TODO: printa tutti i valori e compara con i SET
    //End redis connection
    freeReplyObject(reply);
    redisFree(context);

    return 0;

}

// set to redis using key,value
int setKeyValueStr(char *key, char *value){
    //init connection to redis
    redisContext *context;
    initRedis(&context);

    //set key,value to redis
    auto *reply = (redisReply*)redisCommand(context,"SET %s %s", key, value);
    setChecking(key,reply,context);

    freeReplyObject(reply);
    //set persistent keys
    auto *persist = (redisReply*)redisCommand(context,"PERSIST %s",key);

    if (persist == NULL || persist->type == REDIS_REPLY_ERROR) {
        printf("Error while setting PERSISTENT to key\n");
        redisFree(context);
        return -1;
    }
    freeReplyObject(persist);
    //free space
    redisFree(context);

    return 0;
}

int setKeyValueInt(char *key, int value){
    //init connection to redis
    redisContext *context;
    initRedis(&context);

    //set key,value to redis
    auto *reply = (redisReply*)redisCommand(context,"SET %s %d", key, value);
    setChecking(key,reply,context);

    freeReplyObject(reply);
    //set persistent keys
    auto *persist = (redisReply*)redisCommand(context,"PERSIST %s",key);

    if (persist == NULL || persist->type == REDIS_REPLY_ERROR) {
        printf("Error while setting PERSISTENT to key\n");
        redisFree(context);
        return -1;
    }
    freeReplyObject(persist);
    //free space
    redisFree(context);
    
    return 0;
}


//GET values FROM REDIS USING KEY
char* getStrFromKey(char *key){
    //initialize connection
    redisContext *context;
    initRedis(&context);

    //getting string from redis
    auto reply = (redisReply *) redisCommand(context, "GET %s",key);
    getChecking(key,reply,context);

    char *value;
    if(reply->type == REDIS_REPLY_STRING){
        value = reply->str;
    }else{
        fprintf(stderr,"the object you received is not a str!");
        return nullptr;
    }

    freeReplyObject(reply);
    //end connection;
    redisFree(context);
    return value;
}

int  getIntFromKey(char *key){
    //initialize connection
    redisContext *context;
    initRedis(&context);

    //getting string from redis
    auto reply = (redisReply *) redisCommand(context, "GET %s",key);
    getChecking(key,reply,context);

    int value;
    if(reply->type == REDIS_REPLY_STRING){
        value = strtol(reply->str,nullptr,10);
    }else{
        fprintf(stderr,"the object you received is not a str!");
        return -9999;
    }

    freeReplyObject(reply);
    //end connection;
    redisFree(context);
    return value;
}

//cache on redis a dix commit
int dixCommitToRedis(char *name, char *comment, char **paths){
    /*
    //init connection to redis
    redisContext *context;
    initRedis(&context);

    //store data to redis
    setKeyValueStr((char *)"dixName",name);
    setKeyValueStr((char *)"dixComment",comment);
    i < sizeof(myStrings) / sizeof(myStrings[0])
    for(int i; i< sizeof())

    //finalize connection
    redisFree(context);
    */
}

char **getCharArrayFromRedis(char *key){
    return nullptr;
}