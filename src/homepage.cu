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
     *  6 comandi <p>
     *  - new Name -m (crea il progetto) (-m sta per manuale) <p>
     *  - open Name (apre il progetto) <p>
     *  - del Name  (elimina il progetto) <p>
     *  - view      (visualizza tutti i progetti) <p>
     *  - helpH      (lista dei comandi disponibili) <p>
     *  - exitH      (esce da edix) <p>
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
    else if (strcmp(token, "help") == 0)
        parseHelpH();
    else if (strcmp(token, "exit") == 0)
        parseExitH(env);
    else
        printf(RED "Command not found\n" RESET);


    free(copy);
    return 0;
}

int parseNew(Env *env)
{

    char *token1 = strtok(nullptr, " ");
    char *token2 = strtok(nullptr, " ");

    char *err = strtok(nullptr, " ");


    if (token1 == nullptr || (token2 != nullptr && err != nullptr) )
    {
        handle_error(RED "usage:" RESET " new ProjectName [-m]\n");
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
            handle_error(RED "usage:" RESET " new ProjectName [-m]\n");
        }


        D_PRINT("Ok buon lavoro! ~et\n");
        newP(name, true, env);
        return 0;

    } else if (isValidName(token1))
    {
        D_PRINT("Ok buon lavoro! ~et");
        newP(token1, false, env);
        return 0;
    }

    handle_error(RED "usage:" RESET " new ProjectName [-m]\n");
}
int parseOpen(Env *env)
{
    char *name = strtok(nullptr, " ");

    if (name == nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " open ProjectName\n");
    }
    openP(name, env);

    return 0;
}
int parseDel()
{
    char *name = strtok(nullptr, " ");
    if (name == nullptr || strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " del ProjectName\n");
    }
    delP(name);

    return 0;
}
int parseView()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " view\n");  
    }

    view();
    return 0;
}
int parseHelpH()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " helpH\n");
    }

    helpH();
    return 0;
}
int parseExitH(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error(RED "Usage:" RESET " exitH\n");
    }

    exitH(env);
    return 0;
}


int newP(char *name, bool ask, Env *env)
{
    //TODO: creare il progetto sul db
    //TODO: if ask, chiedi su stdin i settings
    //TODO: cambia working directory
    //TODO: cambia l'env
    D_PRINT("MO SE CREA ER PROGETTO ~et");
    return 0;
}
int openP(char *name, Env *env)
{
    //TODO: controllare se il progetto esiste nel db
    //TODO: in caso cambiare working directory e env
    D_PRINT("MO SE APRE IL PROGETTO ~et");
    return 0;
}
int delP(char *name)
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
    return 0;
}
int helpH()
{
    D_PRINT("Ecco la lista dei comandi da poter eseguire qui sulla homepage:\n\n" 
            "\tnewP\tCrea un nuovo progetto\n" 
            "\topenP\tApri un progetto esistente\n" 
            "\tdelP\tCancella un progetto esistente\n"
            );

    return 0;
}

int banner() 
{
    D_PRINT(BOLD
            "    _/_/_/_/        _/  _/  _/      _/\n"
            "   _/          _/_/_/        _/  _/\n"
            "  _/_/_/    _/    _/  _/      _/\n"
            " _/        _/    _/  _/    _/  _/\n"
            "_/_/_/_/    _/_/_/  _/  _/      _/\n\n"
            RESET
            "Benvenuto su EdiX :). Qui di seguito una lista dei comandi da utilizzare per iniziare a lavore:\n"
            BOLD"  newP" RESET "\tCrea un nuovo progetto\n" 
            BOLD"  openP" RESET "\tApri un progetto esistente\n" 
            BOLD"  delP" RESET "\tCancella un progetto esistente\n");
    
    return 0;
}

int exitH(Env *env)
{
    D_PRINT("Uscita in corso...");
    *env = EXIT;
    return 0;
}