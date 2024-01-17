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

int openConnection(redisContext **context)
{
    *context = redisConnect("localhost", 6379);

    if (*context == nullptr || (*context)->err)
    {
        if (*context)
        {
            fprintf(stderr, RED "Error" RESET "while connecting to redis: %s\n", (*context)->errstr);
            redisFree(*context);
        } else
            fprintf(stderr, RED"Error: " RESET "Unable to initialize connection to redis\n");

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
        handle_error("REDIS Error while executing redis SET command %s\n", name);
    }

    return 0;
}


int getChecking(char *name, redisReply *reply, redisContext *context)
{
    if (reply == nullptr)
    {
        handle_error("REDIS Error while executing GET command %s\n", name);
        freeReplyObject(reply);
        redisFree(context);
        return 1;
    }
    return 0;
}


int projectToRedis(char *name, char *cDate, char *mDate, char *path, int settings)
{

    redisContext *context;
    openConnection(&context);

    //TODO:
    D_PRINT("Adding project name on redis...\n");
    setKeyValueStr((char *) "pName", name);
    setKeyValueStr((char *) "CDate", cDate);
    setKeyValueStr((char *) "MDate", mDate);
    setKeyValueStr((char *) "pPath", path);
    setKeyValueInt((char *) "Settings", settings);

    redisFree(context);

    return 0;
}
int settingsToRedis(int id, char *tup, char *mod_ex, char *comp, u_int tts, char *tpp, bool vcs, char *pName)
{
    //Init redis connection
    redisContext *context;
    openConnection(&context);

    //send settings to redis
    D_PRINT("Adding settings on redis...\n");
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
    redisContext *context;
    openConnection(&context);


    *id = getIntFromKey((char *) "ID");

    *pName = getStrFromKey((char *) "Project");

    *mod_ex = getStrFromKey((char *) "Mod_ex");

    *tts = getIntFromKey((char *) "TTS");

    char *VCS = getStrFromKey((char *) "VCS");
    *vcs = strcmp(VCS, "0");
    free(VCS);

    *comp = getStrFromKey((char *) "COMP");

    *tpp = getStrFromKey((char *) "TPP");

    *tup = getStrFromKey((char *) "TUP");

    redisFree(context);

    return 0;

}


int setKeyValueStr(char *key, char *value)
{

    redisContext *context;
    openConnection(&context);


    auto *reply = (redisReply *) redisCommand(context, "SET %s %s", key, value);
    setChecking(key, reply, context);

    freeReplyObject(reply);

    auto *persist = (redisReply *) redisCommand(context, "PERSIST %s", key);

    if (persist == nullptr || persist->type == REDIS_REPLY_ERROR)
    {
        fprintf(stderr, RED "REDIS Error:" RESET "while setting PERSISTENT to key\n");
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
    openConnection(&context);


    auto *reply = (redisReply *) redisCommand(context, "SET %s %d", key, value);
    setChecking(key, reply, context);

    freeReplyObject(reply);

    auto *persist = (redisReply *) redisCommand(context, "PERSIST %s", key);

    if (persist == nullptr || persist->type == REDIS_REPLY_ERROR)
    {
        fprintf(stderr, RED "REDIS Error:" RESET "while setting PERSISTENT to key\n");
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
    openConnection(&context);


    auto reply = (redisReply *) redisCommand(context, "GET %s", key);
    getChecking(key, reply, context);

    char *value;
    if (reply->type == REDIS_REPLY_STRING)
    {
        value = strdup(reply->str);
    } else
    {
        fprintf(stderr, RED "REDIS Error: " RESET "the object you received is not a str!\n");
        return nullptr;
    }

    freeReplyObject(reply);

    redisFree(context);
    return value;
}
int getIntFromKey(char *key)
{

    redisContext *context;
    openConnection(&context);


    auto reply = (redisReply *) redisCommand(context, "GET %s", key);
    getChecking(key, reply, context);

    int value;
    if (reply->type == REDIS_REPLY_STRING)
    {
        value = (int) strtol(reply->str, nullptr, 10);
    } else
    {
        fprintf(stderr, RED "REDIS Error: " RESET "the object you received is not a str!\n");
        return -9999;
    }

    freeReplyObject(reply);
    redisFree(context);
    return value;
}


int setElementToRedis(char *key, char *value)
{
    redisContext *context;
    openConnection(&context);

    auto *reply = (redisReply *) redisCommand(context, "RPUSH %s %s", key, value);
    if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
    {
        fprintf(stderr,RED "REDIS Error: " RESET " while appending an element to list \n");
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
    openConnection(&context);

    auto *reply = (redisReply *) redisCommand(context, "LRANGE %s 0 -1", key);
    if (reply == nullptr || reply->type != REDIS_REPLY_ARRAY)
    {
        D_PRINT("reply type : %d\n",reply->type);
        fprintf(stderr, RED "REDIS Error: " RESET "Error while retrieving set elements\n");
        freeReplyObject(reply);
        redisFree(context);
        exit(EXIT_FAILURE);
    }

    size_t num_elements = reply->elements;
    char **elements_array = (char **) malloc((num_elements + 1) * sizeof(char *));
    if (elements_array == nullptr)
    {
        fprintf(stderr, RED "REDIS Error: " RESET "Error while allocating array\n");
        freeReplyObject(reply);
        redisFree(context);
        return nullptr;
    }

    for (size_t i = 0; i < num_elements; ++i)
    {
        elements_array[i] = strdup(reply->element[i]->str);
        if (elements_array[i] == nullptr)
        {
            fprintf(stderr, RED "REDIS Error: " RESET "Error while duplicating string\n");
            freeReplyObject(reply);
            free(elements_array);
            redisFree(context);
            return nullptr;
        }
    }

    elements_array[num_elements] = nullptr;
    freeReplyObject(reply);
    redisFree(context);
    return elements_array;
}

int deallocateFromRedis()
{
    redisContext *context;
    openConnection(&context);

    auto *reply = (redisReply *)redisCommand(context ,"FLUSHALL");
    if (reply == nullptr)
    {
        fprintf(stderr, RED "REDIS Error: " RESET "Error while duplicating string\n", context->errstr);
        freeReplyObject(reply);
    }

    freeReplyObject(reply);

    redisFree(context);

    return 0;
}
