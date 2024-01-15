#include <hiredis/hiredis.h>
#include <cstring>
#include "utils.hpp"



//communication with db

int settingsToRedis(int id, char *tup, char *mod_ex, char *comp, u_int tts, char *tpp, bool vcs, char *pName);
int settingsFromRedis(int *id, char **tup, char **mod_ex, char **comp, u_int *tts, char **tpp, bool *vcs, char **pName);
int projectToRedis(char *name, char *cDate, char *mDate, char *path, int settings);

//utils
int dixCommitToRedis(char *name, char *comment, char **paths, char **images);
char **getCharArrayFromRedis(char *key);
int initRedis(redisContext **context);
int setChecking(char *name, redisReply *reply, redisContext *context);
int getChecking(char *name, redisReply *reply, redisContext *context);
int checkRedisService();
int setKeyValueInt(char *key, int value);
int setKeyValueStr(char *key, char *value);
int getIntFromKey(char *key);
char *getStrFromKey(char *key);