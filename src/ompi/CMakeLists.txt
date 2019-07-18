

if(OMPI_FOUND)

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)  
    project(OpenMPI_Example CXX)
 
    set(TARGET ${PROJECT_NAME})

    ppr_blue("Build EXE: ${TARGET}")

    set(SOURCES main.cpp)

    add_executable_mpi(${TARGET} "${SOURCES}")
    # Add a test:
    enable_testing()
    add_test(NAME ${TARGET}_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}_${CMAKE_BUILD_TYPE}/${TARGET})
    #config_test_output(${TARGET}_Test)

  endif()
endif()

