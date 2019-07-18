
cmake_minimum_required(VERSION 3.13)

unset(DO_GTEST CACHE)
unset(DO_TARGET CACHE)
unset(DO_OMPI CACHE)
unset(DO_BOOST CACHE)
unset(DO_PYTHON CACHE)
unset(DO_EIGEN CACHE)

set(CMAKE_LIBRARY_DIR_RELATIVE extern_libs)

set(CMAKE_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/${CMAKE_LIBRARY_DIR_RELATIVE})

option(DO_PYTHON "Build python modules" OFF)
option(DO_GTEST "Use google tests to perform bunch of tests..." OFF)
option(DO_TARGET "Build targets from src dir..." ON)
option(DO_OMPI "Build targets/tests for OpenMPI..." OFF)
option(DO_BOOST "Build targets/tests for Boost..." ON)
option(DO_EIGEN "Build targets/tests for Eigen..." ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/modules_cmake")

include(CMakeCommon)
include(Colors)

set(EXTERN_DIRS "${CMAKE_SOURCE_DIR}/extern")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE $ENV{CMAKE_BUILD_TYPE})
set_build_type(${CMAKE_BUILD_TYPE})


set_property(GLOBAL PROPERTY RESOURCES ${CMAKE_SOURCE_DIR}/resources)


ppr_red(CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


#Default build NAME


if (DO_PYTHON)
    ppr_red("DO_PYTHON")
    include(Python)
    setup_python()
    assert_python()
endif ()

if (DO_OMPI)
    ppr_red("DO_OMPI")
    include(OpenMPI)
    setup_openmpi()
    assert_ompi()
endif ()

if (DO_BOOST)
    ppr_red("DO_BOOST")
    include(Boost)
    setup_boost()
    assert_boost()

endif ()


# Setup Target


if (DO_TARGET)
    message("DO_TARGET is set...")
    add_subdirectory(src)

else (DO_TARGET)
    message("DO_TARGET not set...")

endif (DO_TARGET)


# Setup testing

if (DO_GTEST)
    message("DO_GTEST is set...")
    add_subdirectory(extern_libs/GoogleTest)
    add_subdirectory(test)

else (DO_GTEST)
    message("DO_GTEST not set...")

endif (DO_GTEST)

unset(DO_GTEST CACHE)


file(WRITE "${CMAKE_SOURCE_DIR}/target/cmake_build_type.txt" ${CMAKE_BUILD_TYPE})




