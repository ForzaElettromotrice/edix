#include <hiredis/hiredis.h>
#include <cstring>
#include "utils.hpp"

int initRedis(redisContext **context);
int setChecking(char *name, redisReply *reply, redisContext *context);
int getChecking(char *name, redisReply *reply, redisContext *context);
int settingsToRedis(int id, char *tup, char *mod_ex, char *comp, u_int tts, char *tpp, bool vcs, char  *project)
int settingsFromRedis(int *id, char **tup, char **mod_ex, char **comp, u_int *tts, char **tpp, bool *vcs, char **project);
int checkRedisService();
int setKeyValueInt(char *key, int value);
int setKeyValueStr(char *key, char *value);
int  getIntFromKey(char *key);
char* getStrFromKey(char *key);