
function(assert_python)
    message("PYTHON_FOUND=${PYTHON_FOUND}")
    if (NOT PYTHON_FOUND)
        message(FATAL_ERROR "ERROR: PYTHON is not found")
    endif ()
endfunction()

macro(setup_python)
    unset(PYTHON_FOUND CACHE)

    set(args "${ARGN}")

    list(LENGTH args nargs)

    if (nargs GREATER 0)
        ppr_blue("Python path provided from ${args}")
        set(PYTHON_DIR ${args})
    elseif (EXISTS ${CMAKE_LIBRARY_PATH}/Python27)
        ppr_blue("Python path provided from ${CMAKE_LIBRARY_PATH}/Python27")
        set(PYTHON_DIR ${CMAKE_LIBRARY_PATH}/Python27)
    else ()
        ppr_blue("Python path provided from python-config --prefix...")
        execute_process(COMMAND python-config --prefix OUTPUT_VARIABLE PYTHON_DIR)
    endif ()


    string(STRIP ${PYTHON_DIR} PYTHON_DIR)

    message("PYTHON_DIR = ${PYTHON_DIR}")

    execute_process(COMMAND ${PYTHON_DIR}/bin/python -c "from sysconfig import get_paths; print(get_paths()['stdlib'])" OUTPUT_VARIABLE PYTHON_LIB_PATH)
    execute_process(COMMAND ${PYTHON_DIR}/bin/python -c "from sysconfig import get_paths; print(get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_PATH)
    execute_process(COMMAND ${PYTHON_DIR}/bin/python -c "from sysconfig import get_paths; print(get_paths()['purelib'])" OUTPUT_VARIABLE PYTHON_SITEPACKAGES)

    execute_process(COMMAND ${PYTHON_DIR}/bin/python-config --libs OUTPUT_VARIABLE PYTHON_LIBS)
    execute_process(COMMAND ${PYTHON_DIR}/bin/python-config --cflags OUTPUT_VARIABLE PYTHON_COMPILE_FLAGS)
    execute_process(COMMAND ${PYTHON_DIR}/bin/python-config --ldflags OUTPUT_VARIABLE PYTHON_LINK_FLAGS)

    string(STRIP "${PYTHON_LIB_PATH}" PYTHON_LIB_PATH)
    string(STRIP "${PYTHON_INCLUDE_PATH}" PYTHON_INCLUDE_PATH)
    string(STRIP "${PYTHON_SITEPACKAGES}" PYTHON_SITEPACKAGES)
    string(STRIP "${PYTHON_LIBS}" PYTHON_LIBS)
    string(STRIP "${PYTHON_COMPILE_FLAGS}" PYTHON_COMPILE_FLAGS)
    string(STRIP "${PYTHON_LINK_FLAGS}" PYTHON_LINK_FLAGS)

    set(PYTHON_EXECUTABLE ${PYTHON_DIR}/bin/python)

    set(ENV{LD_LIBRARY_PATH} "$ENV{LD_LIBRARY_PATH}:${PYTHON_LIB_PATH}")

    MESSAGE(ENV{LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH})

    MESSAGE(PYTHON_DIR=${PYTHON_DIR})
    MESSAGE(PYTHON_LIB_PATH=${PYTHON_LIB_PATH})
    MESSAGE(PYTHON_INCLUDE_PATH=${PYTHON_INCLUDE_PATH})
    MESSAGE(PYTHON_SITEPACKAGES=${PYTHON_SITEPACKAGES})
    MESSAGE(PYTHON_LIBS=${PYTHON_LIBS})
    MESSAGE(PYTHON_COMPILE_FLAGS=${PYTHON_COMPILE_FLAGS})
    MESSAGE(PYTHON_LINK_FLAGS=${PYTHON_LINK_FLAGS})
    MESSAGE(PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE})

    #link_directories(${PYTHON_LIB_PATH})
    #include_directories(${PYTHON_INCLUDE_PATH})

    option(PYTHON_FOUND "Set flag that PYTHON is found now" ON)

    message("PYTHON_FOUND=${PYTHON_FOUND}")
endmacro()


function(_add_module_python_ _NAME)
    assert_python()

    get_property(_TARGET_SUPPORTS_SHARED_LIBS
            GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
    option(PYTHON_ENABLE_MODULE_${_NAME} "Add module ${_NAME}" TRUE)
    option(PYTHON_MODULE_${_NAME}_BUILD_SHARED
            "Add module ${_NAME} shared" ${_TARGET_SUPPORTS_SHARED_LIBS})

    # Mark these options as advanced
    mark_as_advanced(PYTHON_ENABLE_MODULE_${_NAME}
            PYTHON_MODULE_${_NAME}_BUILD_SHARED)

    if (PYTHON_ENABLE_MODULE_${_NAME})

        if (PYTHON_MODULE_${_NAME}_BUILD_SHARED)
            set(PY_MODULE_TYPE MODULE)
        else ()
            set(PY_MODULE_TYPE STATIC)
            set_property(GLOBAL APPEND PROPERTY PY_STATIC_MODULES_LIST ${_NAME})

        endif ()

        set_property(GLOBAL APPEND PROPERTY PY_MODULES_LIST ${_NAME})
        add_library(${_NAME} ${PY_MODULE_TYPE} ${ARGN})
        #    target_link_libraries(${_NAME} ${PYTHON_LIBRARIES})

        if (PYTHON_MODULE_${_NAME}_BUILD_SHARED)
            set_target_properties(${_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
            if (WIN32 AND NOT CYGWIN)
                set_target_properties(${_NAME} PROPERTIES SUFFIX ".pyd")
            endif ()
        endif ()

    endif ()

    if (DEFINED PY_MODULE_TYPE)
        set(PY_MODULE_TYPE ${PY_MODULE_TYPE} PARENT_SCOPE)
    else ()
        set(PY_MODULE_TYPE MODULE PARENT_SCOPE)
    endif ()

endfunction()

macro(link_python TARGET)
    # Attach directories
    target_link_directories(${TARGET} PRIVATE ${PYTHON_LIB_PATH})
    target_include_directories(${TARGET} PRIVATE ${PYTHON_INCLUDE_PATH})

    target_compile_options(${TARGET} PRIVATE ${PYTHON_COMPILE_FLAGS})
    target_link_libraries(${TARGET} ${PYTHON_LINK_FLAGS})

    set_target_properties(${TARGET} PROPERTIES FOLDER ${CMAKE_LIBRARY_PATH})

    target_compile_definitions(${TARGET} PRIVATE
            LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
            )
endmacro()

macro(add_module_python TARGET SOURCES)
    ppr_blue("add_executable_python : TARGET=${TARGET}; SOURCES=${SOURCES}...")

    assert_python()

    if (NOT DEFINED PROJECT_NAME)
        message("Python project not defined. Set PROJECT(${TARGET})...")
        PROJECT(${TARGET})
    endif ()


    _add_module_python_(${TARGET} ${SOURCES})
    #if(DEFINED FAKE_TARGET_OMPI)
    #  add_dependencies(${TARGET} ${FAKE_TARGET_OMPI})
    #endif()

    if (${PY_MODULE_TYPE} EQUAL MODULE)
        config_lib_output(${TARGET})
    elseif (${PY_MODULE_TYPE} EQUAL STATUS)
        config_arch_output(${TARGET})
    endif ()

    # Attach directories
    target_link_directories(${TARGET} PRIVATE ${PYTHON_LIB_PATH})
    target_include_directories(${TARGET} PRIVATE ${PYTHON_INCLUDE_PATH})

    target_compile_options(${TARGET} PRIVATE ${PYTHON_COMPILE_FLAGS})
    target_link_libraries(${TARGET} ${PYTHON_LINK_FLAGS})

    set_target_properties(${TARGET} PROPERTIES FOLDER ${CMAKE_LIBRARY_PATH})

    target_compile_definitions(${TARGET} PRIVATE
            LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
            )

endmacro()

