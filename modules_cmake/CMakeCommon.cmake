

macro(link_resources TARGET)

    get_runtime_output_dir(${TARGET})
    message("RUNTIME_OUTPUT_DIR = ${RUNTIME_OUTPUT_DIR}")

    set(source "${CMAKE_SOURCE_DIR}/resources")
    set(destination "${RUNTIME_OUTPUT_DIR}/resources")

    MESSAGE("source = ${source}")
    MESSAGE("destination = ${destination}")

    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${source} ${destination}
            DEPENDS ${destination}
    )

endmacro()


function(set_build_type)

    unset(CMAKE_BUILD_TYPE CACHE)

    if (DEFINED ARGN1)
        set(default_build_type "${ARGN1}")
    else ()
        set(default_build_type "Release")
    endif ()

    if (NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
        set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
                STRING "Choose the type of build." FORCE)
        # Set the possible values of build NAME for modules_cmake-gui
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif ()


endfunction()

macro(get_runtime_output_dir TARGET)


    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        get_target_property(RUNTIME_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY_DEBUG)
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        get_target_property(RUNTIME_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY_RELEASE)
    elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        get_target_property(RUNTIME_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL)
    elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        get_target_property(RUNTIME_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO)
    else ()
        get_target_property(RUNTIME_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY)
    endif ()


endmacro()

macro(get_test_output_dir TARGET)

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        get_target_property(TEST_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY_DEBUG)
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        get_target_property(TEST_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY_RELEASE)
    elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        get_target_property(TEST_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL)
    elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        get_target_property(TEST_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO)
    else ()
        get_target_property(TEST_OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY)
    endif ()

endmacro()


macro(get_library_output_dir TARGET)

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        get_target_property(LIBRARY_OUTPUT_DIR ${TARGET} LIBRARY_OUTPUT_DIRECTORY_DEBUG)
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        get_target_property(LIBRARY_OUTPUT_DIR ${TARGET} LIBRARY_OUTPUT_DIRECTORY_RELEASE)
    elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        get_target_property(LIBRARY_OUTPUT_DIR ${TARGET} LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL)
    elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        get_target_property(LIBRARY_OUTPUT_DIR ${TARGET} LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO)
    else ()
        get_target_property(LIBRARY_OUTPUT_DIR ${TARGET} LIBRARY_OUTPUT_DIRECTORY)
    endif ()

endmacro()

macro(config_arch_output TARGET)
    set(_DIR "${CMAKE_SOURCE_DIR}/target")
    set_target_properties(${TARGET} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${_DIR}/arc")
    set_target_properties(${TARGET} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${_DIR}/debug/arc")
    set_target_properties(${TARGET} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${_DIR}/release/arc")
    set_target_properties(${TARGET} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_DIR}/relwithdebinfo/arc")
    set_target_properties(${TARGET} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${_DIR}/minsizerel/arc")
endmacro(config_arch_output)


macro(config_lib_output TARGET)
    set(_DIR "${CMAKE_SOURCE_DIR}/target")
    set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_DIR}/lib")
    set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${_DIR}/debug/lib")
    set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${_DIR}/release/lib")
    set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_DIR}/relwithdebinfo/lib")
    set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${_DIR}/minsizerel/lib")
endmacro(config_lib_output)


macro(config_bin_output TARGET)
    set(_DIR "${CMAKE_SOURCE_DIR}/target")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${_DIR}/bin")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${_DIR}/debug/bin")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_DIR}/release/bin")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_DIR}/relwithdebinfo/bin")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${_DIR}/minsizerel/bin")
endmacro(config_bin_output)

macro(config_test_output TARGET)
    set(_DIR "${CMAKE_SOURCE_DIR}/target")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${_DIR}")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${_DIR}/debug/test")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_DIR}/release/test")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_DIR}/relwithdebinfo/test")
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${_DIR}/minsizerel/test")
endmacro(config_test_output)

macro(config_output TARGET)

    config_bin_output(${TARGET})
    config_test_output(${TARGET})
    config_lib_output(${TARGET})
    config_arch_output(${TARGET})

endmacro()


function(JOIN OUT GLUE)
    SET(TEMP)
    foreach (arg IN ITEMS ${ARGN})
        STRING(STRIP "${arg}" arg)
        SET(TEMP "${TEMP} ${GLUE}${arg}")
    endforeach ()
    STRING(STRIP "${TEMP}" TEMP)
    SET(${OUT} "${TEMP}" PARENT_SCOPE)
endfunction()

# External libraries which need to be built.
#add_subdirectory ( ${CMAKE_SOURCE_DIR}/extern_libs/GoogleTest extern_libs/GoogleTest )
#set_target_properties(gtest PROPERTIES FOLDER extern_libs)
#set_target_properties(gtest_main PROPERTIES FOLDER extern_libs)
#set_target_properties(gmock PROPERTIES FOLDER extern_libs)
#set_target_properties(gmock_main PROPERTIES FOLDER extern_libs)


#NEVER SET ROOT, it will remove all paths to ./modules_cmake
#set(CMAKE_ROOT ${CMAKE_SOURCE_DIR})


#list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/embree2")
#list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/sdl2")
#list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/glew")
#list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/glm")


#message("install: DIRECTORY ${CMAKE_SOURCE_DIR}/extern_libs/OpenMPI DESTINATION ${CMAKE_BINARY_DIR}/extern_libs/OpenMPI")

#install(DIRECTORY ${CMAKE_SOURCE_DIR}/extern_libs/OpenMPI DESTINATION ${CMAKE_BINARY_DIR}/extern_libs/OpenMPI )
#executes after doing: make install

#if(NOT EXISTS ${CMAKE_BINARY_DIR}/${OpenMPI_DIR})
#  add_custom_command(TARGET ${FAKE_TARGET_OMPI} PRE_BUILD COMMAND cp ${CMAKE_SOURCE_DIR}/${OpenMPI_DIR} ${RUNTIME_OUTPUT_DIRECTORY}/${OpenMPI_DIR} -Rfn)
#endif()


# add_custom_command(TARGET ${TARGET}
#                   POST_BUILD COMMAND
#                   /bin/cp ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_INIT} ${_DIR}/__init__.py
# )
