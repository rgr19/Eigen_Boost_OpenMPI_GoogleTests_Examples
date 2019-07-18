
function(OutSourceBuild)

message("PROJECT_NAME = ${PROJECT_NAME}")
message("...CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message("...CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")

option(IS_SOURCE_BUILD ON)

set(DEFAULT_OUT_OF_SOURCE_FOLDER "cmake-build")

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    message(WARNING "In-source builds not allowed. CMake will now be run with arguments:
        cmake -H. -B${DEFAULT_OUT_OF_SOURCE_FOLDER}
")


    # Run CMake with out of source flag
    execute_process(
      COMMAND ${CMAKE_COMMAND} -H. -B${DEFAULT_OUT_OF_SOURCE_FOLDER} -DIS_SOURCE_BUILD=OFF
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    

    # Cause fatal error to stop the script from further execution
    message(FATAL_ERROR "CMake has been ran to create an out of source build.
This error prevents CMake from running an in-source build.")
  
endif ()

return()

endfunction()
