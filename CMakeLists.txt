cmake_minimum_required(VERSION 3.7)
project(vec_add_svm LANGUAGES CXX)
find_package(OpenCL QUIET)
subdirs(src)
