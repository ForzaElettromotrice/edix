cmake_minimum_required(VERSION 3.27)
project(edix CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 20)


find_package(OpenMP REQUIRED)

set(SOURCE_FILES
        src/dbutils/pgutils.cpp
        src/dbutils/pgutils.hpp
        src/env/homepage.cpp
        src/env/homepage.hpp
        src/env/project.cpp
        src/env/project.hpp
        src/env/settings.cpp
        src/env/settings.hpp
        src/main.cpp
        src/main.hpp
        src/dbutils/rdutils.cpp
        src/dbutils/rdutils.hpp
        src/functions/functions.cuh
        src/functions/grayscale.cu
        src/functions/blur.cu
        src/functions/imgutils.cpp
        src/functions/colorfilter.cu
        src/functions/overlap.cu
        src/functions/composition.cu
        src/functions/composition.cuh
        src/functions/compression.cu
        src/functions/compression.cuh
        src/functions/blur.cuh
        src/functions/overlap.cuh
        src/functions/imgutils.hpp
        src/functions/scale.cu
        src/functions/scale.cuh
)

set(SOURCE_FILES_TEST
        test/main.hpp
        test/main.cpp
        test/testFunc.cpp
        test/testFunc.hpp
        src/functions/functions.cuh
        src/functions/grayscale.cu
        src/functions/blur.cu
        src/functions/imgutils.cpp
        src/functions/colorfilter.cu
        src/functions/overlap.cu
        src/functions/composition.cu
        src/functions/composition.cuh
        src/functions/compression.cu
        src/functions/compression.cuh
        src/functions/blur.cuh
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
        src/functions/scale.cu
        src/functions/scale.cuh
)

# Aggiungi tutte le sorgenti del progetto
add_executable(edix ${SOURCE_FILES})

add_executable(testx ${SOURCE_FILES_TEST})

# Aggiungi le opzioni di compilazione per CUDA
target_compile_options(edix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
target_compile_options(testx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)

# Link alle librerie necessarie
target_link_libraries(edix PRIVATE -lhiredis -lpq -ljpeg -lpng)
target_link_libraries(edix PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(testx PRIVATE -lhiredis -lpq -ljpeg -lpng)
target_link_libraries(testx PRIVATE OpenMP::OpenMP_CXX)
