cmake_minimum_required(VERSION 3.27)
project(edix CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 20)


find_package(OpenMP REQUIRED)

set(SOURCE_FILES
        src/functions/scale.cuh
        src/functions/composition.cuh
        src/functions/compression.cuh
        src/functions/blur.cuh
        src/functions/grayscale.cu
        src/functions/blur.cu
        src/functions/overlap.cuh
        src/functions/colorfilter.cu
        src/functions/overlap.cu
        src/functions/composition.cu
        src/functions/compression.cu
        src/functions/scale.cu
        src/dbutils/pgutils.cpp
        src/dbutils/pgutils.hpp
        src/env/homepage.cpp
        src/env/homepage.hpp
        src/env/project.cpp
        src/env/project.hpp
        src/env/settings.cpp
        src/env/settings.hpp
        src/env/linenoise.c
        src/env/linenoise.h
        src/main.cpp
        src/main.hpp
        src/dbutils/rdutils.cpp
        src/dbutils/rdutils.hpp
        src/functions/imgutils.cpp
        src/functions/imgutils.hpp
        src/functions/parser.cpp
        src/functions/parser.hpp
        src/uitils.cpp
)

set(SOURCE_FILES_TEST
        src/functions/scale.cuh
        src/functions/composition.cuh
        src/functions/compression.cuh
        src/functions/blur.cuh
        src/functions/grayscale.cu
        src/functions/blur.cu
        src/functions/colorfilter.cu
        src/functions/overlap.cu
        src/functions/composition.cu
        src/functions/compression.cu
        src/functions/scale.cu
        test/main.hpp
        test/main.cpp
        test/testFunc.cu
        test/testFunc.cuh
        src/functions/imgutils.cpp
        src/dbutils/pgutils.cpp
        src/dbutils/pgutils.hpp
        src/env/homepage.cpp
        src/env/homepage.hpp
        src/env/project.cpp
        src/env/project.hpp
        src/env/settings.cpp
        src/env/settings.hpp
        src/dbutils/rdutils.cpp
        src/dbutils/rdutils.hpp
        src/functions/imgutils.hpp
        src/functions/parser.cpp
        src/functions/parser.hpp
        src/uitils.cpp
)

# Aggiungi tutte le sorgenti del progetto
add_executable(edix ${SOURCE_FILES})

add_executable(testx ${SOURCE_FILES_TEST})

# Aggiungi le opzioni di compilazione per CUDA
target_compile_options(edix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
target_compile_options(testx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
#target_compile_options(edix PRIVATE -g)

# Link alle librerie necessarie
target_link_libraries(edix PRIVATE -lhiredis -lpq -ljpeg -lpng)
target_link_libraries(edix PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(testx PRIVATE -lhiredis -lpq -ljpeg -lpng)
target_link_libraries(testx PRIVATE OpenMP::OpenMP_CXX)
