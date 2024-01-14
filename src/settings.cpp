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
     *  - view     (visualizza settings correnti)
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
        handle_error("usage" BOLD ITALIC " mv fromPath toPath\n" RESET);
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


int set(char *name, char *value)
{
    //TODO
    return 0;
}
int helpS()
{
    printf("Ecco la lista dei settings:\n\n"
            RED BOLD " MODE" RESET  ITALIC"  Modalità di esecuzione\n\t" RESET " * Programmata: Effettua le modifiche apportate a un'immagine dopo un certo periodo di tempo" BOLD "\n\t * Immediata" RESET "(default): Le modifiche vengono effettuate subito\n\n" 
            RED BOLD " TTS"  RESET  ITALIC"   Ogni quanti minuti salva su disco\n\t" RESET BOLD " * 5m" RESET "(default): Ogni 5m il progetto viene salvato su disco\n\n"
            RED BOLD " VCS"  RESET  ITALIC"   Version Control System\n\t * On: Puoi tenere traccia nel tempo delle modifiche che si effettuano sulle immagini\n\t" RESET BOLD " * Off" RESET "(default): Non viene tenuta traccia delle modifiche\n\n"
            RED BOLD " COMP" RESET  ITALIC"  Formato di compressione dell'immagine\n\t" RESET BOLD " * PPM" RESET "(default)\n\t * JPEG\n\t * PNG\n\n"
            RED BOLD " TPP"  RESET  ITALIC"   Tecnologia di parallelismo\n\t" RESET " * CUDA: Elaborazione parallela su unità di elaborazione grafica (GPU)\n\t * OMP: Elaborazione parallela su sistemi condivisi di memoria\n\t *" BOLD " Seriale" RESET "(default): Non viene effettuata alcuna ottimizzazione\n\n"
            RED BOLD " TUP"  RESET  ITALIC"   Tecnologia di upscaling\n\t" RESET BOLD " * Bilineare" RESET "(default)\n\t * Bicubica\n\n");
            
    return 0;
}
int exitS(Env *env)
{
    //TODO
    return 0;
}