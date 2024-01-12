cmake_minimum_required(VERSION 3.27)
project(edix CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES
        src/dbutils.cpp
        src/dbutils.h
        src/homepage.cpp
        src/homepage.h
        src/project.cpp
        src/project.h
        src/settings.cpp
        src/settings.h
        src/main.cpp
        src/main.h
)

# Aggiungi tutte le sorgenti del progetto
add_executable(edix ${SOURCE_FILES})

# Aggiungi le opzioni di compilazione per CUDA
target_compile_options(edix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)

# Link alle librerie necessarie
target_link_libraries(edix PRIVATE -lhiredis -lpq)
