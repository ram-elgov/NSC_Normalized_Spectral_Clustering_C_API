cmake_minimum_required(VERSION 3.22)
project(NSC___Normalized_Spectral_Clustering_C_API C)

set(CMAKE_C_STANDARD 90)

add_executable(NSC___Normalized_Spectral_Clustering_C_API spkmeans.c spkmeans.h spkmeansmodule.c)