cmake_minimum_required(VERSION 3.27)
project(edix CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES
        src/dbutils.cpp
        src/dbutils.hpp
        src/homepage.cpp
        src/homepage.hpp
        src/project.cpp
        src/project.hpp
        src/settings.cpp
        src/settings.hpp
        src/main.cpp
        src/main.hpp
)

# Aggiungi tutte le sorgenti del progetto
add_executable(edix ${SOURCE_FILES})

# Aggiungi le opzioni di compilazione per CUDA
target_compile_options(edix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)

# Link alle librerie necessarie
target_link_libraries(edix PRIVATE -lhiredis -lpq)
