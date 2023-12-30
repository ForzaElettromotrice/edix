//
// Created by f3m on 30/12/23.
//

#include "homepage.h"


bool isValidName(char *word)
{
    for (size_t i = 0; i < strlen(word) - 1; ++i)
    {
        if (!isalnum(word[i]) && word[i] != '_')
        {
            //TODO: fare i print carini
            D_PRINT("Invalid character %c", word[i]);
            return false;
        }
    }

    return true;
}
bool isValidFlag(char *flag)
{
    return flag[0] == '-' && flag[1] == 'm';
}

int parseHome(char *line, Env *env)
{
    /**
     *  4 comandi
     *  - new Name -m (crea il progetto) (-m sta per manuale)
     *  - open Name (apre il progetto)
     *  - del Name  (elimina il progetto)
     *  - view      (visualizza tutti i progetti)
     */

    char *copy = strdup(line);
    char *token = strtok(copy, " ");

    if (strcmp(token, "new") == 0)
        parseNew(env);
    else if (strcmp(token, "open") == 0)
    {

    } else if (strcmp(token, "del") == 0)
    {

    } else if (strcmp(token, "view") == 0)
    {

    } else
    {
        printf("Command not found\n");
    }
    free(copy);

    return 0;
}

int parseNew(Env *env)
{
    char *token1 = strtok(nullptr, " ");
    char *token2 = strtok(nullptr, " ");

    char *err = strtok(nullptr, " ");

    if (token2 != nullptr && err != nullptr)
    {
        printf("usage: new ProjectName [-m]\n");
        return 1;
    }

    if (token2 != nullptr)
    {
        char *name;

        if (isValidName(token1) && isValidFlag(token2))
        {
            name = token1;
        } else if (isValidName(token2) && isValidFlag(token1))
        {
            name = token2;
        } else
        {
            printf("usage: new ProjectName [-m]\n");
            return 1;
        }


        D_PRINT("Ok buon lavoro! ~et\n");
        newProject(name, true);
        *env = PROJECT;
        return 0;

    } else if (isValidName(token1))
    {
        D_PRINT("Ok buon lavoro! ~et");
        newProject(token1, false);
        *env = PROJECT;
        return 0;
    }

    printf("usage: new ProjectName [-m]\n");
    return 1;
}

int newProject(char *name, bool ask)
{
    //TODO: creare il progetto sul db
    //TODO: if ask, chiedi su stdin i settings
    //TODO: cambia working directory

    return 0;
}