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
bool isValidFlag(const char *flag)
{
    return flag[0] == '-' && flag[1] == 'm';
}

int parseHome(char *line, Env *env)
{
    /**
     *  4 comandi <p>
     *  - new Name -m (crea il progetto) (-m sta per manuale) <p>
     *  - open Name (apre il progetto) <p>
     *  - del Name  (elimina il progetto) <p>
     *  - view      (visualizza tutti i progetti) <p>
     */

    char *copy = strdup(line);
    char *token = strtok(copy, " ");


    if (strcmp(token, "new") == 0)
        parseNew(env);
    else if (strcmp(token, "open") == 0)
        parseOpen(env);
    else if (strcmp(token, "del") == 0)
        parseDel();
    else if (strcmp(token, "view") == 0)
        parseView();
    else
        printf("Command not found\n");


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
int parseOpen(Env *env)
{
    char *name = strtok(nullptr, " ");
    if (strtok(nullptr, " ") != nullptr)
    {
        printf("Usage: open ProjectName\n");
        return 1;
    }

    openProject(name, env);

    return 0;
}
int parseDel()
{
    char *name = strtok(nullptr, " ");
    if (strtok(nullptr, " ") != nullptr)
    {
        printf("Usage: del ProjectName\n");
        return 1;
    }

    delProject(name);

    return 0;
}
int parseView()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        printf("Usage: view");
        return 1;
    }

    view();
    return 0;
}


int newProject(char *name, bool ask)
{
    //TODO: creare il progetto sul db
    //TODO: if ask, chiedi su stdin i settings
    //TODO: cambia working directory
    D_PRINT("MO SE CREA ER PROGETTO ~et");
    return 0;
}
int openProject(char *name, Env *env)
{
    //TODO: controllare se il progetto esiste nel db
    //TODO: in caso cambiare working directory e env
    D_PRINT("MO SE APRE IL PROGETTO ~et");
    return 0;
}
int delProject(char *name)
{
    //TODO: controllare se esiste nel db
    //TODO: in caso cancellare ogni cosa

    D_PRINT("MO TE CANCELLO LA VITA ~et");

    return 0;
}
int view()
{
    //TODO: leggi dal db tutti i progetti e le varie info

    D_PRINT("ECCHETE LA VIEW ~et");
}