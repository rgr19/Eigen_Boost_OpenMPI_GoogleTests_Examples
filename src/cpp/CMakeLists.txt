project(Cpp_Example_Simple LANGUAGES C CXX)
ppr_blue("Build EXECUTABLE: ${PROJECT_NAME}...")


add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS ON)

# create simple lib

ppr_blue("Build LIBRARY: ${PROJECT_NAME}_Lib...")
add_library(${PROJECT_NAME}_Lib simple_lib.cpp simple_lib.hpp)
config_lib_output(${PROJECT_NAME}_Lib)
set_target_properties(${PROJECT_NAME}_Lib PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS ON)


include(CheckIPOSupported)
check_ipo_supported(RESULT result)
if (result)
    set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_Lib)

config_bin_output(${PROJECT_NAME})

###

project(Cpp_Example_Maybe_Boost LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.13)

message("Build EXE: ${PROJECT_NAME}")

set(BOOST_LIBS
        python27
        numpy27
        )

set(SOURCES
        MUtil.h
        matrix_operations.cpp
        )

if (DO_BOOST)
    ppr_blue("Build BOOST EXE: ${PROJECT_NAME}...")
    add_executable_boost(${PROJECT_NAME} "${SOURCES}" "${BOOST_LIBS}")
else ()
    ppr_blue("Build EXE: ${PROJECT_NAME}...")
    add_executable(${PROJECT_NAME} ${SOURCES})
    config_bin_output(${PROJECT_NAME})
endif ()

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS ON)

if (DO_PYTHON)
    link_python(${PROJECT_NAME})
endif ()

link_resources(${PROJECT_NAME})

