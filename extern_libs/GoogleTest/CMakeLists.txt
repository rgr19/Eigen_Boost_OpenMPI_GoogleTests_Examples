

if(DO_GTEST)

    cmake_minimum_required(VERSION 3.13)
    #set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_BINARY_DIR}/cmake-build)

    message("Extern :: GOOGLE TEST...")
    message("... Project name = ${PROJECT_NAME}")
    message("... PROJECT_SRC_DIR = ${PROJECT_SRC_DIR}")
    message("... CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")
    message("... CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
    message("... PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
    message("... PROJECT_BINARY_DIR=${PROJECT_BINARY_DIR}")
    MESSAGE("... CMAKE_ROOT_DIR=${CMAKE_ROOT_DIR}")
  

    configure_file(CMakeLists.txt.gtest.in googletest-download/CMakeLists.txt)
   

    message("Execute process: ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR}  . ")
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    
    if(result)
      message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()

    message("Execute process: ${CMAKE_COMMAND} --build . ")
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
      message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()
    
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    
 
    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                   ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                     EXCLUDE_FROM_ALL)
  


   
    # The gtest/gtest_main targets carry header search path
    # dependencies automatically when using CMake 2.8.11 or
    # later. Otherwise we have to add them here ourselves.


    message("Gtest source dir: ${gtest_SOURCE_DIR}")

    if (CMAKE_VERSION VERSION_LESS 2.8.11)
      include_directories("${gtest_SOURCE_DIR}/include")
    endif()


endif(DO_GTEST)


