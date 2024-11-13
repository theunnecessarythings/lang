cmake_minimum_required(VERSION 3.20.0)
project(lang LANGUAGES CXX C)

# Set up the project
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=lld")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=lld")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=lld")
add_compile_options(-fexceptions)

# Setup LLVM and MLIR
set(PREFIX "/mnt/ubuntu/home/sreeraj/Documents/llvm-project/build")
set(LLVM_DIR "${PREFIX}/lib/cmake/llvm")
set(MLIR_DIR "${PREFIX}/lib/cmake/mlir")
set(LLVM_EXTERNAL_LIT "${PREFIX}/bin/llvm-lit")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
find_package(MLIR REQUIRED CONFIG)


list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(dialect)

# Main project as a library
add_library(liblang STATIC 
    src/parser.cpp
    src/ast.cpp
    src/MLIRGen.cpp
    src/LangGen.cpp
    src/analyzer.cpp
    src/lower_to_llvm.cpp
    src/lower_to_affine.cpp
    src/literal_cast_pass.cpp
    src/json_dumper.cpp)  
target_include_directories(liblang PUBLIC include/)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
    Lang)
add_llvm_executable(lang main.cpp)

llvm_update_compile_flags(lang)
target_link_libraries(lang PRIVATE ${LIBS} liblang)

target_link_libraries(liblang PRIVATE ${LIBS})


# Catch2 for testing
Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 
)
FetchContent_MakeAvailable(Catch2)

# Tests
add_executable(tests tests/test_lexer.cpp tests/test_parser.cpp tests/test_analyzer.cpp)
target_include_directories(tests PRIVATE include/)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain liblang)
