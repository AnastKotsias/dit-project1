cmake_minimum_required(VERSION 3.20)
project(image_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        add_compile_options(-g -fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

find_package(OpenCV REQUIRED)

add_executable(search
    src/main.cpp
    src/image_features.cpp
    src/vocabulary.cpp
    src/dataset.cpp
    src/pipeline.cpp
    src/search.cpp
)

target_include_directories(search PRIVATE include)
target_link_libraries(search PRIVATE ${OpenCV_LIBS})