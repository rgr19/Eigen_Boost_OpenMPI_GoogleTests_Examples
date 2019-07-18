  
message("When calling setup_boost(BOOST_LIBS, ...), skip part '-lboost_' at each library in list BOOST_LIBS...")

function(assert_boost)
    message("BOOST_FOUND=${BOOST_FOUND}")
    if (NOT BOOST_FOUND)
        message(FATAL_ERROR "ERROR: Boost is not found")
    endif ()
endfunction()

macro(setup_boost)
    unset(BOOST_FOUND CACHE)

    if (DEFINED ARGN1)
        set(BOOST_DIR ${ARGN1})
    else ()
        set(BOOST_DIR ${CMAKE_LIBRARY_PATH}/Boost)
    endif ()

    message("BOOST_DIR = ${BOOST_DIR}")


    set(BOOST_LIB_PATH ${BOOST_DIR}/lib)
    set(BOOST_INCLUDE_PATH ${BOOST_DIR}/include)

    set(ENV{LD_LIBRARY_PATH} "$ENV{LD_LIBRARY_PATH}:${BOOST_LIB_PATH}")

    MESSAGE(ENV{OPAL_PREFIX=$ENV{OPAL_PREFIX})
    MESSAGE(ENV{LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH})

    option(BOOST_FOUND "Set flag that BOOST is found now" ON)

    message("BOOST_FOUND=${BOOST_FOUND}")
endmacro()

function(target_link_libraries_boost TARGET)

    assert_boost()

    JOIN(BOOST_LIBS -lboost_ "${ARGN}")

    MESSAGE(BOOST_LIBS=${BOOST_LIBS})

    target_link_libraries(${TARGET} ${BOOST_LIBS})

endfunction()

macro(add_executable_boost TARGET SOURCES BOOST_LIBS)

    if (DEFINED ARGN)
        set(BOOST_LIBS ${ARGN})
    else ()
        set(BOOST_LIBS)
    endif ()


    ppr_blue("add_executable_boost : TARGET=${TARGET}; SOURCES=${SOURCES}...")

    assert_boost()

    if (NOT DEFINED PROJECT_NAME)
        message("BOOST project not defined. Set PROJECT(${TARGET})...")
        PROJECT(${TARGET})
    endif ()


    add_executable(${TARGET} ${SOURCES})
    #if(DEFINED FAKE_TARGET_BOOST)
    #  add_dependencies(${TARGET} ${FAKE_TARGET_BOOST})
    #endif()

    # Attach directories
    target_link_directories(${TARGET} PRIVATE ${BOOST_LIB_PATH})
    target_include_directories(${TARGET} PRIVATE ${BOOST_INCLUDE_PATH})
    target_link_libraries_boost(${TARGET} ${BOOST_LIBS})


    set_target_properties(${TARGET} PROPERTIES FOLDER ${CMAKE_LIBRARY_PATH})

    target_compile_definitions(${TARGET} PRIVATE
            OPAL_PREFIX=$ENV{OPAL_PREFIX}
            LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
            WITH_BOOST=1
            )

    config_bin_output(${TARGET})


endmacro()

macro(add_module_boost_python TARGET SOURCES)

    if (DEFINED ARGN)
        set(BOOST_LIBS ${ARGN})
    else ()
        set(BOOST_LIBS)
    endif ()


    ppr_blue("add_module_boost_python : CCSD=${CMAKE_CURRENT_SOURCE_DIR} : TARGET=${TARGET}; SOURCES=${SOURCES}...")

    assert_boost()
    assert_python()

    if (NOT DEFINED PROJECT_NAME)
        message("BOOST project not defined. Set PROJECT(${TARGET})...")
        PROJECT(${TARGET})
    endif ()

    add_module_python(${TARGET} ${SOURCES})
    #if(DEFINED FAKE_TARGET_BOOST)
    #  add_dependencies(${TARGET} ${FAKE_TARGET_BOOST})
    #endif()

    # Attach directories
    target_link_directories(${TARGET} PRIVATE ${BOOST_LIB_PATH})
    target_include_directories(${TARGET} PRIVATE ${BOOST_INCLUDE_PATH})
    target_link_libraries_boost(${TARGET} ${BOOST_LIBS})

    set_target_properties(${TARGET} PROPERTIES FOLDER ${CMAKE_LIBRARY_PATH})

    target_compile_definitions(${TARGET} PRIVATE
            LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
            WITH_BOOST=1
            WITH_BOOST_PYTHON=1
            )

endmacro()


macro(add_python_init TARGET PYTHON_INI)

    get_target_property(RUNTIME_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE})


endmacro()


macro(add_executable_boost_ompi TARGET SOURCES)

    if (DEFINED ARGN)
        set(BOOST_LIBS ${ARGN})
    else ()
        set(BOOST_LIBS)
    endif ()

    assert_boost()
    assert_ompi()

    ppr_blue("BUILD boost + ompi EXE with BOOST-LIBS: ${BOOST_LIBS}...")

    if (NOT DEFINED PROJECT_NAME)
        message("BOOST + OMPI project not defined. Set PROJECT(${TARGET})...")
        PROJECT(${TARGET})
    endif ()

    add_executable(${TARGET} ${SOURCES})
    #if(DEFINED FAKE_TARGET_OMPI)
    #  add_dependencies(${TARGET} ${FAKE_TARGET_OMPI})
    #endif()

    # Attach directories
    target_link_directories(${TARGET} PRIVATE ${MPI_LIB_PATH} ${BOOST_LIB_PATH})
    target_include_directories(${TARGET} PRIVATE ${MPI_INCLUDE_PATH} ${BOOST_INCLUDE_PATH})

    # Attach openmpi flags
    target_compile_options(${TARGET} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    target_link_libraries(${TARGET} ${MPI_CXX_LINK_FLAGS})
    target_link_libraries_boost(${TARGET} ${BOOST_LIBS} mpi serialization)

    set_target_properties(${TARGET} PROPERTIES FOLDER ${CMAKE_LIBRARY_PATH})

    if (${CMAKE_BUILD_TYPE} EQ "Release")
        message("BOOST NO DEBUG INFO...")
        target_compile_definitions(${TARGET} PRIVATE
                OPAL_PREFIX=$ENV{OPAL_PREFIX}
                LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
                WITH_BOOS=1
                WITH_BOOST_MPI=1
                NDEBUG=1
                BOOST_UBLAS_NDEBUG=1
                )
    else ()
        target_compile_definitions(${TARGET} PRIVATE
                OPAL_PREFIX=$ENV{OPAL_PREFIX}
                LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
                WITH_BOOS=1
                WITH_BOOST_MPI=1
                )
    endif ()
    config_bin_output(${TARGET})


endmacro()

macro(post_build_run_boost_mpi TARGET NP)


    add_custom_command(TARGET ${TARGET}
            POST_BUILD COMMAND
            echo OPAL_PREFIX=$ENV{OPAL_PREFIX}
            LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
            ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NP} $<TARGET_FILE:${TARGET}>)

    add_custom_command(TARGET ${TARGET}
            POST_BUILD COMMAND
            OPAL_PREFIX=$ENV{OPAL_PREFIX}
            LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
            ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NP} $<TARGET_FILE:${TARGET}> || true)


endmacro()

