cmake_minimum_required(VERSION 3.20.0)
project(lang LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_library(liblang STATIC src/parser.cpp src/render.cpp)  
target_include_directories(liblang PUBLIC include/)

add_executable(lang main.cpp)
target_include_directories(lang PRIVATE include/)  
target_link_libraries(lang PRIVATE liblang)

Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 
)
FetchContent_MakeAvailable(Catch2)

add_executable(tests tests/test_lexer.cpp tests/test_parser.cpp)
target_include_directories(tests PRIVATE include/)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain liblang)
