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

int checkRedisService()
{

    redisContext *context;
    context = redisConnect("localhost", 6379);

    if (context == nullptr || (context)->err)
    {
        handle_error("Redis is not running\n");
    }
    return 0;
}


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


int projectToRedis(char *name, char *cDate, char *mDate, char *path, int settings)
{
    //initialize conn with redis
    redisContext *context;
    initRedis(&context);

    //send project to redis
    setKeyValueStr((char *) "Name", name);
    setKeyValueStr((char *) "CDate", cDate);
    setKeyValueStr((char *) "MDate", mDate);
    setKeyValueStr((char *) "pPath", path);
    setKeyValueInt((char *) "Settings", settings);

    //end redis connection
    redisFree(context);

    return 0;
}
int settingsToRedis(int id, char *tup, char *mod_ex, char *comp, u_int tts, char *tpp, bool vcs, char *pName)
{
    //Init redis connection
    redisContext *context;
    initRedis(&context);

    //send settings to redis
    setKeyValueInt((char *) "ID", id);
    setKeyValueStr((char *) "Project", pName);
    setKeyValueStr((char *) "Mod_ex", mod_ex);
    setKeyValueInt((char *) "TTS", (int) tts);
    setKeyValueInt((char *) "VCS", vcs);
    setKeyValueStr((char *) "COMP", comp);
    setKeyValueStr((char *) "TPP", tpp);
    setKeyValueStr((char *) "TUP", tup);

    //End redis connection
    redisFree(context);

    return 0;
}
int settingsFromRedis(int *id, char **tup, char **mod_ex, char **comp, u_int *tts, char **tpp, bool *vcs, char **pName)
{
    //init redis connection
    redisContext *context;
    initRedis(&context);

    //TODO: controllare i return di getChecking...
    //TODO: poi sta funzione non me piace, ce deve esse un modo piu intelligente di farla

    // Recupera i dati dalle chiavi

    *id = getIntFromKey((char *) "ID");

    *pName = getStrFromKey((char *) "Project");

    *mod_ex = getStrFromKey((char *) "Mod_ex");

    *tts = getIntFromKey((char *) "TTS");

    auto *reply = (redisReply *) redisCommand(context, "GET VCS");
    getChecking((char *) "VCS", reply, context);
    *vcs = reply->integer == 1;

    *comp = getStrFromKey((char *) "COMP");

    *tpp = getStrFromKey((char *) "TPP");

    *tup = getStrFromKey((char *) "TUP");

    //TODO: printa tutti i valori e compara con i SET
    //End redis connection
    freeReplyObject(reply);
    redisFree(context);

    return 0;

}


int setKeyValueStr(char *key, char *value)
{

    redisContext *context;
    initRedis(&context);


    auto *reply = (redisReply *) redisCommand(context, "SET %s %s", key, value);
    setChecking(key, reply, context);

    freeReplyObject(reply);

    auto *persist = (redisReply *) redisCommand(context, "PERSIST %s", key);

    if (persist == nullptr || persist->type == REDIS_REPLY_ERROR)
    {
        printf("Error while setting PERSISTENT to key\n");
        redisFree(context);
        return -1;
    }
    freeReplyObject(persist);

    redisFree(context);

    return 0;
}
int setKeyValueInt(char *key, int value)
{

    redisContext *context;
    initRedis(&context);


    auto *reply = (redisReply *) redisCommand(context, "SET %s %d", key, value);
    setChecking(key, reply, context);

    freeReplyObject(reply);

    auto *persist = (redisReply *) redisCommand(context, "PERSIST %s", key);

    if (persist == nullptr || persist->type == REDIS_REPLY_ERROR)
    {
        printf("Error while setting PERSISTENT to key\n");
        redisFree(context);
        return -1;
    }
    freeReplyObject(persist);

    redisFree(context);

    return 0;
}


char *getStrFromKey(char *key)
{

    redisContext *context;
    initRedis(&context);


    auto reply = (redisReply *) redisCommand(context, "GET %s", key);
    getChecking(key, reply, context);

    char *value;
    if (reply->type == REDIS_REPLY_STRING)
    {
        value = strdup(reply->str);
    } else
    {
        fprintf(stderr, "the object you received is not a str!");
        return nullptr;
    }

    freeReplyObject(reply);

    redisFree(context);
    return value;
}
int getIntFromKey(char *key)
{

    redisContext *context;
    initRedis(&context);


    auto reply = (redisReply *) redisCommand(context, "GET %s", key);
    getChecking(key, reply, context);

    int value;
    if (reply->type == REDIS_REPLY_STRING)
    {
        value = (int) strtol(reply->str, nullptr, 10);
    } else
    {
        fprintf(stderr, "the object you received is not a str!");
        return -9999;
    }

    freeReplyObject(reply);
    //end connection;
    redisFree(context);
    return value;
}


//cache on redis a dix commit
int dixCommitToRedis(char *name, char *comment, char **paths, char **images)
{

    redisContext *context;
    initRedis(&context);

    setElementToRedis((char *) "dixNames", name);
    setElementToRedis((char *) "dixComments", comment);

    char *pathname;
    for (int i = 0; paths[i] != nullptr; i++)
    {
        char path[32] = "Paths";
        pathname = strdup(name);
        if (pathname == nullptr)
        {
            handle_error("Error while duplicating string");
            return 1;
        }
        strcat(pathname, path);
        setElementToRedis(pathname, paths[i]);
    }

    char *imgname;
    for (int i = 0; images[i] != nullptr; i++)
    {
        char img[32] = "Images";
        imgname = strdup(name);
        if (imgname == nullptr)
        {
            handle_error("Error while duplicating string");
            return 1;
        }
        strcat(imgname, img);
        setElementToRedis(imgname, images[i]);

    }

    //finalize connection
    getCharArrayFromRedis((char *) "dixNames");
    getCharArrayFromRedis((char *) "dixComments");
    getCharArrayFromRedis(pathname);
    getCharArrayFromRedis(imgname);

    free(pathname);
    free(imgname);
    redisFree(context);

    return 0;
}

int setElementToRedis(char *key, char *value)
{
    redisContext *context;
    initRedis(&context);

    auto *reply = (redisReply *) redisCommand(context, "SADD %s %s", key, value);
    if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
    {
        printf("Error while appending an element to list \n");
        freeReplyObject(reply);
        redisFree(context);
        return 1;
    }
    freeReplyObject(reply);

    redisFree(context);
    return 0;
}

char **getCharArrayFromRedis(char *key)
{
    redisContext *context;
    initRedis(&context);

    auto *reply = (redisReply *) redisCommand(context, "SMEMBERS %s", key);
    if (reply == NULL || reply->type != REDIS_REPLY_ARRAY)
    {
        printf("Error while retrieving set elements\n");
        freeReplyObject(reply);
        redisFree(context);
        exit(EXIT_FAILURE);
    }


    size_t num_elements = reply->elements;
    char **elements_array = (char **) malloc(num_elements * sizeof(char *));
    if (elements_array == NULL)
    {
        printf("Error while allocating array\n");
        freeReplyObject(reply);
        redisFree(context);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < num_elements; ++i)
    {
        elements_array[i] = strdup(reply->element[i]->str);
        if (elements_array[i] == NULL)
        {
            printf("Error while duplicating string\n");
            freeReplyObject(reply);
            free(elements_array);
            redisFree(context);
            exit(EXIT_FAILURE);
        }
    }

    redisFree(context);
    //TODO: la lista che ritorna deve avere come ultimo valore nullptr, così ci posso iterare sopra senza sapere la lunghezza
    return elements_array;
}