//
// Created by f3m on 12/01/24.
//

#include "settings.hpp"

int parseSettings(char *line, Env *env)
{
    /**
     *  3 comandi                                   <p>
     *  - set       (setta un' impostazione)        <p>
     *  - help     (lista dei comandi disponibili) <p>
     *  - listH     (visualizza settings correnti)
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
    else if (strcmp(token, "list") == 0)
        parseListS();
    else
        printf(RED "Command not found\n" RESET);


    free(copy);
    return 0;
}

int parseSet()
{
    char *name = strtok(nullptr, " ");
    char *value = strtok(nullptr, " ");


    if ((name == nullptr || value == nullptr) || strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " set Key Value\n" RESET);
        return 1;
    }


    set(name, value);

    return 0;
}
int parseHelpS()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " help\n" RESET);
        return 1;
    }

    helpS();

    return 0;
}
int parseExitS(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " exit\n" RESET);
        return 1;
    }

    exitS(env);

    return 0;
}
int parseListS()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        E_Print("usage" BOLD ITALIC " list\n" RESET);
        return 1;
    }

    listS();

    return 0;
}

int set(char *name, char *value)
{
    if (strcmp(name, "TTS") == 0)
    {
        uint val = strtol(value, nullptr, 10);
        if (val > 5)
            setKeyValueInt("TTS", (int) val);
        else
        {
            E_Print("Valore non valido!\n");
            return 1;
        }
    } else if (strcmp(name, "Backup") == 0)
    {

        if (strcmp(value, "On") == 0)
            setKeyValueStr("Backup", (char *) "true");
        else if (strcmp(value, "Off") == 0)
            setKeyValueStr("Backup", (char *) "false");
        else
        {
            E_Print("Valore non valido!\n");
            return 1;
        }
    } else if (strcmp(name, "COMP") == 0)
    {
        if (strcmp(value, "PPM") == 0 or strcmp(value, "JPEG") == 0 or strcmp(value, "PNG") == 0)
            setKeyValueStr("COMP", value);
        else
        {
            E_Print("Valore non valido!\n");
            return 1;
        }
    } else if (strcmp(name, "TPP") == 0)
    {
        if (strcmp(value, "CUDA") == 0 or strcmp(value, "OMP") == 0 or strcmp(value, "Serial") == 0)
            setKeyValueStr("TPP", value);
        else
        {
            E_Print("Valore non valido!\n");
            return 1;
        }
    } else if (strcmp(name, "TUP") == 0)
    {
        if (strcmp(value, "Bilinear") == 0 or strcmp(value, "Bicubic") == 0)
            setKeyValueStr("TUP", value);
        else
        {
            E_Print("Valore non valido!\n");
            return 1;
        }
    } else
    {
        E_Print("Valore non valido!\n");
        return 1;
    }
    return 0;
}
int helpS()
{
    printf("Ecco la lista di comandi da poter utilizzare all'interno dei settings:\n\n"
           YELLOW BOLD "  list" RESET "\t\t\tStampa i settings impostati\n"
           YELLOW BOLD "  set "  RESET UNDERLINE "KEY" RESET " " UNDERLINE "VALUE" RESET "\t\tImposta il setting KEY a VALUE\n"
           YELLOW BOLD "  exit" RESET "\t\t\tEsci dai settings\n\n");

    return 0;
}
int exitS(Env *env)
{
    D_Print("Uscita dai settings...\n");
    *env = PROJECT;
    return 0;
}
int listS()
{
    char *tup;
    int tts;
    char *tpp;
    bool backup;

    settingsFromRedis(&tup, &tts, &tpp, &backup);

    printf("Ecco una lista dettagliata dei settings:\n\n"
           RED BOLD " TTS  "   RESET  ITALIC"   Ogni quante istruzioni salva su disco\n\t" RESET BOLD "    * 5" RESET "(default): Ogni 5 istruzioni il progetto viene salvato su disco\n\n"
           RED BOLD " Backup"  RESET  ITALIC"  Version Control System\n\t    * On: Puoi tenere traccia nel tempo delle modifiche che si effettuano sulle immagini\n\t" RESET BOLD "    * Off" RESET "(default): Non viene tenuta traccia delle modifiche\n\n"
           RED BOLD " TPP  "   RESET  ITALIC"   Tecnologia di parallelismo\n\t" RESET "    * CUDA: Elaborazione parallela su unità di elaborazione grafica (GPU)\n\t    * OMP: Elaborazione parallela su sistemi condivisi di memoria\n\t    *" BOLD " Serial" RESET "(default): Non viene effettuata alcuna ottimizzazione\n\n"
           RED BOLD " TUP  "   RESET  ITALIC"   Tecnologia di upscaling\n\t" RESET BOLD "    * Bilinear" RESET "(default)\n\t    * Bicubic\n");

    puts("----------------------------------------------------------------------------------------------------------------------------------------------------------------");

    printf(BOLD YELLOW "IMPOSTAZIONI ATTUALI:\n" RESET
           BOLD "\n * " GREEN "TUP    " RESET BOLD "==>" RESET " [%s]"
           BOLD "\n * " GREEN "TTS    " RESET BOLD "==>" RESET " [%u]"
           BOLD "\n * " GREEN "TPP    " RESET BOLD "==>" RESET " [%s]"
           BOLD "\n * " GREEN "Backup " RESET BOLD "==>" RESET " [%s]\n\n", tup, tts, tpp, backup ? "On" : "Off");


    return 0;
}