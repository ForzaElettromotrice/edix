#ifndef REDIS_RDUTILS_HPP
#define REDIS_RDUTILS_HPP

#include "rdutils.hpp"

int openConnection(redisContext **context)
{
    *context = redisConnect("localhost", 6379);

    if (*context == nullptr || (*context)->err)
    {
        if (*context)
        {
            E_Print(RED "Error" RESET "while connecting to redis: %s\n", (*context)->errstr);
            redisFree(*context);
        } else
            E_Print(RED"Error: " RESET "Unable to initialize connection to redis\n");

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
        E_Print("Redis is not running\n");
        return 1;
    }
    return 0;
}


int
settingsFromRedis(int *id, char **tup, char **comp, u_int *tts, char **tpp, bool *backup, char **pName)
{
    redisContext *context;
    openConnection(&context);


    *id = getIntFromKey((char *) "ID");
    *pName = getStrFromKey((char *) "Project");
    *tts = getIntFromKey((char *) "TTS");
    char *Backup = getStrFromKey((char *) "Backup");
    *backup = strcmp(Backup, "0");
    free(Backup);
    *comp = getStrFromKey((char *) "COMP");
    *tpp = getStrFromKey((char *) "TPP");
    *tup = getStrFromKey((char *) "TUP");

    redisFree(context);

    return 0;

}
int settingsToRedis(int id, char *tup, char *comp, u_int tts, char *tpp, bool backup, char *pName)
{
    redisContext *context;
    openConnection(&context);

    D_Print("Adding settings on redis...\n");
    setKeyValueInt((char *) "ID", id);
    setKeyValueStr((char *) "Project", pName);
    setKeyValueInt((char *) "TTS", (int) tts);
    setKeyValueInt((char *) "Backup", backup);
    setKeyValueStr((char *) "COMP", comp);
    setKeyValueStr((char *) "TPP", tpp);
    setKeyValueStr((char *) "TUP", tup);

    redisFree(context);

    return 0;
}
int projectToRedis(char *name, char *cDate, char *mDate, char *path, int settings)
{

    redisContext *context;
    openConnection(&context);

    //TODO: mettere i D_Print
    D_Print("Adding project name on redis...\n");
    setKeyValueStr((char *) "pName", name);
    setKeyValueStr((char *) "CDate", cDate);
    setKeyValueStr((char *) "MDate", mDate);
    setKeyValueStr((char *) "pPath", path);
    setKeyValueInt((char *) "Settings", settings);

    redisFree(context);

    return 0;
}


int checkResponse(char *name, redisReply *reply, redisContext *context)
{
    if (reply == nullptr)
    {
        freeReplyObject(reply);
        redisFree(context);
        E_Print("REDIS Error while executing command %s\n", name);
        return 1;
    }
    return 0;
}
int setKeyValueStr(char *key, char *value)
{

    redisContext *context;
    openConnection(&context);


    auto *reply = (redisReply *) redisCommand(context, "SET %s %s", key, value);
    checkResponse(key, reply, context);

    freeReplyObject(reply);

    redisFree(context);

    return 0;
}
int setKeyValueInt(char *key, int value)
{

    redisContext *context;
    openConnection(&context);


    auto *reply = (redisReply *) redisCommand(context, "SET %s %d", key, value);
    checkResponse(key, reply, context);

    freeReplyObject(reply);

    redisFree(context);

    return 0;
}
int setElementToRedis(char *key, char *value)
{
    redisContext *context;
    openConnection(&context);

    auto *reply = (redisReply *) redisCommand(context, "RPUSH %s %s", key, value);
    if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
    {
        E_Print(RED "REDIS Error: " RESET " while appending an element to list \n");
        freeReplyObject(reply);
        redisFree(context);
        return 1;
    }
    freeReplyObject(reply);

    redisFree(context);
    return 0;
}
char *getStrFromKey(char *key)
{

    redisContext *context;
    openConnection(&context);


    auto reply = (redisReply *) redisCommand(context, "GET %s", key);
    checkResponse(key, reply, context);

    char *value;
    if (reply->type == REDIS_REPLY_STRING)
    {
        value = strdup(reply->str);
    } else
    {
        E_Print(RED "REDIS Error: " RESET "the object you received is not a str!\n");
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
    checkResponse(key, reply, context);

    int value;
    if (reply->type == REDIS_REPLY_STRING)
    {
        value = (int) strtol(reply->str, nullptr, 10);
    } else
    {
        E_Print(RED "REDIS Error: " RESET "the object you received is not a str!\n");
        return -9999;
    }

    freeReplyObject(reply);
    redisFree(context);
    return value;
}
char **getCharArrayFromRedis(char *key)
{
    redisContext *context;
    openConnection(&context);

    auto *reply = (redisReply *) redisCommand(context, "LRANGE %s 0 -1", key);
    if (reply == nullptr || reply->type != REDIS_REPLY_ARRAY)
    {
        D_Print("reply type : %d\n", reply->type);
        E_Print(RED "REDIS Error: " RESET "Error while retrieving set elements\n");
        freeReplyObject(reply);
        redisFree(context);
        exit(EXIT_FAILURE);
    }

    size_t num_elements = reply->elements;
    char **elements_array = (char **) malloc((num_elements + 1) * sizeof(char *));
    if (elements_array == nullptr)
    {
        E_Print(RED "REDIS Error: " RESET "Error while allocating array\n");
        freeReplyObject(reply);
        redisFree(context);
        return nullptr;
    }

    for (size_t i = 0; i < num_elements; ++i)
    {
        elements_array[i] = strdup(reply->element[i]->str);
        if (elements_array[i] == nullptr)
        {
            E_Print(RED "REDIS Error: " RESET "Error while duplicating string\n");
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


int removeKeyFromRedis(char *key)
{
    redisContext *context;
    openConnection(&context);

    auto *reply = (redisReply *) redisCommand(context, "DEL %s", key);
    if (reply == nullptr)
    {
        E_Print(RED "REDIS Error: " RESET "Error while deleting key: %s\n", key);
        return 1;
    }


    freeReplyObject(reply);
    redisFree(context);

    return 0;
}
int delDixFromRedis()
{
    redisContext *context;
    openConnection(&context);
    char *key = (char *) malloc(256 * sizeof(char));

    char **dixs = getCharArrayFromRedis((char *) "dixNames");
    for (int i = 0; dixs[i] != nullptr; ++i)
    {
        sprintf(key, "%sPaths", dixs[i]);
        removeKeyFromRedis(key);
        sprintf(key, "%sImages", dixs[i]);
        removeKeyFromRedis(key);
        free(dixs[i]);
    }
    free(dixs);

    removeKeyFromRedis((char *) "dixNames");
    removeKeyFromRedis((char *) "dixComments");

    return 0;
}
int deallocateFromRedis()
{
    redisContext *context;
    openConnection(&context);

    delDixFromRedis();
    removeKeyFromRedis((char *) "pName");
    removeKeyFromRedis((char *) "CDate");
    removeKeyFromRedis((char *) "MDate");
    removeKeyFromRedis((char *) "pPath");
    removeKeyFromRedis((char *) "Settings");

    removeKeyFromRedis((char *) "ID");
    removeKeyFromRedis((char *) "Project");
    removeKeyFromRedis((char *) "TTS");
    removeKeyFromRedis((char *) "Backup");
    removeKeyFromRedis((char *) "COMP");
    removeKeyFromRedis((char *) "TPP");
    removeKeyFromRedis((char *) "TUP");

    redisFree(context);

    return 0;
}

#endif //REDIS_RDUTILS_HPP