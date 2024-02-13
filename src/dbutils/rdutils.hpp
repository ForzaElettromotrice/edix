#include <hiredis/hiredis.h>
#include <cstring>
#include "../utils.hpp"


int openConnection(redisContext **context);
int checkRedisService();


int settingsFromRedis(int *id, char **tup, int *tts, char **tpp, bool *backup, char **pName);
int settingsToRedis(char *tup, int tts, char *tpp, bool backup);
int projectToRedis(char *name, char *cDate, char *mDate, char *path);


//utils
int checkResponse(const char *name, redisReply *reply, redisContext *context);

int setKeyValueInt(const char *key, int value);
int setKeyValueStr(const char *key, char *value);
int setElementToRedis(const char *key, char *value);
int getIntFromKey(const char *key);
char *getStrFromKey(const char *key);
char **getCharArrayFromRedis(const char *key);

int delDixFromRedis();
int removeKeyFromRedis(const char *key);
int deallocateFromRedis();
