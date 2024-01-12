cmake_minimum_required(VERSION 3.27)
project(edix CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(edix
        src/main.cu
        src/homepage.cu
        src/project.cu
        src/temp.cu
        src/dbutils.cu)
