cmake_minimum_required(VERSION 3.20)
project(image_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(search-hp
    src/main.cpp
    src/image_features.cpp
)

target_include_directories(search-hp PRIVATE include)
target_link_libraries(search-hp PRIVATE ${OpenCV_LIBS})