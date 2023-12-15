cmake_minimum_required(VERSION 3.27)
project(photoeditor CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(photoeditor src/main.cu)

set_target_properties(photoeditor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
