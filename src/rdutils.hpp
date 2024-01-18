#include <hiredis/hiredis.h>
#include <cstring>
#include "utils.hpp"



//communication with db

int settingsToRedis(int id, char *tup, char *mode, char *comp, u_int tts, char *tpp, bool backup, char *pName);
int
settingsFromRedis(int *id, char **tup, char **mode, char **comp, u_int *tts, char **tpp, bool *backup, char **pName);
int projectToRedis(char *name, char *cDate, char *mDate, char *path, int settings);

//utils
//int dixCommitToRedis(char *name, char *comment, char **paths, char **images);
int setElementToRedis(char *key, char *value);
char **getCharArrayFromRedis(char *key);
int openConnection(redisContext **context);
int removeKeyFromRedis(char *key);
int deallocateFromRedis();
int setChecking(char *name, redisReply *reply, redisContext *context);
int getChecking(char *name, redisReply *reply, redisContext *context);
int checkRedisService();
int setKeyValueInt(char *key, int value);
int setKeyValueStr(char *key, char *value);
int getIntFromKey(char *key);
char *getStrFromKey(char *key);
