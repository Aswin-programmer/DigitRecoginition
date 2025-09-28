cmake_minimum_required(VERSION 3.10.0)
project(DigitRecoginition VERSION 0.1.0 LANGUAGES C CXX)

add_subdirectory(DEPENDENCIES/MyLib)

add_executable(DigitRecoginition main.cpp)

target_link_libraries(DigitRecoginition PRIVATE MyLib)

target_compile_definitions(DigitRecoginition PRIVATE
    RESOURCES_PATH="${CMAKE_SOURCE_DIR}/ASSETS/"#Never leave the space between the = sign!.
)