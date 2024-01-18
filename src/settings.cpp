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


    if ((name != nullptr && value != nullptr) || strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " set Key Value\n" RESET);
    }


    set(name, value);

    return 0;
}
int parseHelpS()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " help\n" RESET);
    }

    helpS();

    return 0;
}
int parseExitS(Env *env)
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " exit\n" RESET);
    }

    exitS(env);

    return 0;
}
int parseListS()
{
    if (strtok(nullptr, " ") != nullptr)
    {
        handle_error("usage" BOLD ITALIC " list\n" RESET);
    }

    listS();

    return 0;
}

int set(char *name, char *value)
{
    if (strcmp(name, "Modex") == 0)
    {
        if (strcmp(value, "Immediate") == 0 or strcmp(value, "Programmed") == 0)
            setKeyValueStr((char *) "Modex", value);
        else
        {
            handle_error("Valore non valido!\n");
        }
    } else if (strcmp(name, "TTS") == 0)
    {
        uint val = strtol(value, nullptr, 10);
        if (val > 5)
            setKeyValueInt((char *) "TTS", (int) val);
        else
        {
            handle_error("Valore non valido!\n");
        }
    } else if (strcmp(name, "Backup") == 0)
    {

        if (strcmp(value, "On") == 0)
            setKeyValueStr((char *) "Backup", (char *) "true");
        else if (strcmp(value, "Off") == 0)
            setKeyValueStr((char *) "Backup", (char *) "false");
        else
        {
            handle_error("Valore non valido!\n");
        }
    } else if (strcmp(name, "COMP") == 0)
    {
        if (strcmp(value, "PPM") == 0 or strcmp(value, "JPEG") == 0 or strcmp(value, "PNG") == 0)
            setKeyValueStr((char *) "COMP", value);
        else
        {
            handle_error("Valore non valido!\n");
        }
    } else if (strcmp(name, "TPP") == 0)
    {
        if (strcmp(value, "CUDA") == 0 or strcmp(value, "OMP") == 0 or strcmp(value, "Serial") == 0)
            setKeyValueStr((char *) "TPP", value);
        else
        {
            handle_error("Valore non valido!\n");
        }
    } else if (strcmp(name, "TUP") == 0)
    {
        if (strcmp(value, "Bilinear") == 0 or strcmp(value, "Bicubic") == 0)
            setKeyValueStr((char *) "TUP", value);
        else
        {
            handle_error("Valore non valido!\n");
        }
    } else
    {
        handle_error("Valore non valido!\n");
    }
    return 0;
}
int helpS()
{
    printf("Ecco la lista dei settings:\n\n"
           RED BOLD " MODE" RESET  ITALIC"  Modalità di esecuzione\n\t" RESET " * Programmed: Effettua le modifiche apportate a un'immagine dopo un certo periodo di tempo" BOLD "\n\t * Immediate" RESET "(default): Le modifiche vengono effettuate subito\n\n"
           RED BOLD " TTS"  RESET  ITALIC"   Ogni quanti minuti salva su disco\n\t" RESET BOLD " * 5" RESET "(default): Ogni 5m il progetto viene salvato su disco\n\n"
           RED BOLD " Backup"  RESET  ITALIC"   Version Control System\n\t * On: Puoi tenere traccia nel tempo delle modifiche che si effettuano sulle immagini\n\t" RESET BOLD " * Off" RESET "(default): Non viene tenuta traccia delle modifiche\n\n"
           RED BOLD " COMP" RESET  ITALIC"  Formato di compressione dell'immagine\n\t" RESET BOLD " * PPM" RESET "(default)\n\t * JPEG\n\t * PNG\n\n"
           RED BOLD " TPP"  RESET  ITALIC"   Tecnologia di parallelismo\n\t" RESET " * CUDA: Elaborazione parallela su unità di elaborazione grafica (GPU)\n\t * OMP: Elaborazione parallela su sistemi condivisi di memoria\n\t *" BOLD " Serial" RESET "(default): Non viene effettuata alcuna ottimizzazione\n\n"
           RED BOLD " TUP"  RESET  ITALIC"   Tecnologia di upscaling\n\t" RESET BOLD " * Bilinear" RESET "(default)\n\t * Bicubic\n\n");

    return 0;
}
int exitS(Env *env)
{
    //TODO
    return 0;
}
int listS()
{
    int id;
    char *pName;
    char *modex;
    uint tts;
    bool backup;
    char *comp;
    char *tpp;
    char *tup;

    settingsFromRedis(&id, &tup, &modex, &comp, &tts, &tpp, &backup, &pName);

    //TODO: da farli piu carini
    printf(BOLD YELLOW "IMPOSTAZIONI ATTUALI:" RESET "\n\tTUP = %s"
           "\n\tMODEX = %s"
           "\n\tCOMP = %s"
           "\n\tTTS = %u"
           "\n\tTPP = %s"
           "\n\tBackup = %d\n", tup, modex, comp, tts, tpp, backup);

    return 0;
}