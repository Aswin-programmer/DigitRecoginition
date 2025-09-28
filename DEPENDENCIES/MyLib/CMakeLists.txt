

add_library(MyLib STATIC 
    CSVLoader.h CSVLoader.cpp
    NumCPP.h NumCPP.cpp NumCPP.inl
)

target_include_directories(MyLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})