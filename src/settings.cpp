//
// Created by f3m on 12/01/24.
//

#include "settings.h"

int parseSettings(char *line, Env *env)
{
    /**
     *  3 comandi                                   <p>
     *  - set       (setta un' impostazione)        <p>
     *  - help     (lista dei comandi disponibili) <p>
     *  - exit     (esce dal progetto)             <p>
     */
    char *copy = strdup(line);
    char *token = strtok(copy, " ");


    if (strcmp(token, "set") == 0)
        parseSet();
    else if (strcmp(token, "help") == 0)
        parseHelpS();
    else if (strcmp(token, "exit") == 0)
        parseExitS(env);
    else
        printf(RED "Command not found\n" RESET);


    free(copy);
    return 0;
}

int parseSet()
{
    char *name = strtok(nullptr, " ");
    char *value = strtok(nullptr, " ");


    if ((name != nullptr && value != nullptr) || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " mv fromPath toPath\n");
    }


    set(name, value);

    return 0;
}
int parseHelpS()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " help\n");
    }

    helpS();

    return 0;
}
int parseExitS(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " exit\n");
    }

    exitS(env);

    return 0;
}


int set(char *name, char *value)
{
    //TODO
    return 0;
}
int helpS()
{
    //TODO
    return 0;
}
int exitS(Env *env)
{
    //TODO
    return 0;
}