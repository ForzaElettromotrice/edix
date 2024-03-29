#ifndef EDIX_MAIN_HPP
#define EDIX_MAIN_HPP

#include <iostream>
#include <uv.h>
#include <chrono>
#include "env/homepage.hpp"
#include "env/project.hpp"
#include "env/settings.hpp"
#include "env/linenoise.h"
#include "dbutils/pgutils.hpp"
#include "functions/compression.cuh"
#include "../test/testFunc.cuh"


//MAIN
int inputLoop(Env env);
int print_prompt(Env env);


#endif
