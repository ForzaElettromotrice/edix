#include <hiredis/hiredis.h>
#include <cstring>
#include "../utils.hpp"


int openConnection(redisContext **context);
int checkRedisService();


int
settingsFromRedis(int *id, char **tup, char **mode, char **comp, u_int *tts, char **tpp, bool *backup, char **pName);
int settingsToRedis(int id, char *tup, char *mode, char *comp, u_int tts, char *tpp, bool backup, char *pName);
int projectToRedis(char *name, char *cDate, char *mDate, char *path, int settings);


//utils
int checkResponse(char *name, redisReply *reply, redisContext *context);

int setKeyValueInt(char *key, int value);
int setKeyValueStr(char *key, char *value);
int setElementToRedis(char *key, char *value);
int getIntFromKey(char *key);
char *getStrFromKey(char *key);
char **getCharArrayFromRedis(char *key);

int delDixFromRedis();
int removeKeyFromRedis(char *key);
int deallocateFromRedis();
