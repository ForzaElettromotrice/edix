//
// Created by f3m on 12/01/24.
//

#ifndef EDIX_SETTINGS_HPP
#define EDIX_SETTINGS_HPP

#include <iostream>
#include <cstring>
#include "utils.hpp"

//PARSERS
int parseSettings(char *line, Env *env);
int parseSet();
int parseHelpS();
int parseExitS(Env *env);

//COMMANDS
int set(char *name, char *value);
int helpS();
int exitS(Env *env);

#endif //EDIX_SETTINGS_HPP
