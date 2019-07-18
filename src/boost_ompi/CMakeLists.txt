Project(Boost_OpenMPI_Example VERSION 1.0.0 LANGUAGES C CXX)


if(BOOST_FOUND AND OMPI_FOUND)

    ##

    set(TARGET boost_mpi.boost-ompi)

    ppr_blue("Build EXE: ${TARGET}")

    set(SOURCES
            main.boost.ompi.cpp
            )


    add_executable_boost_ompi(${TARGET} "${SOURCES}" )


    ##

    set(TARGET boost_mpi_fielsystem.boost-ompi)
    ppr_blue("Build EXE: ${TARGET}")
    set(SOURCES
            main.filesystem.ompi.cpp
    )
    set(BOOST_LIBS
        filesystem
    )
    add_executable_boost_ompi(${TARGET} "${SOURCES}" ${BOOST_LIBS})

    ##

    set(TARGET boost_mpi_p2p_blocking.boost-ompi)
    ppr_blue("Build EXE: ${TARGET}")
    set(SOURCES
            main.p2p.blocking.ompi.cpp
            )

    add_executable_boost_ompi(${TARGET} "${SOURCES}" )

    ##

    set(TARGET boost_mpi_p2p_nonblocking.boost-ompi)
    ppr_blue("Build EXE: ${TARGET}")
    set(SOURCES
            main.p2p.non-blocking.ompi.cpp
            )

    add_executable_boost_ompi(${TARGET} "${SOURCES}" )

    ##

    set(TARGET boost_threading_mpi.boost-ompi)
    ppr_blue("Build EXE: ${TARGET}")
    set(SOURCES
            main.boost.threading.ompi.cpp
            )

    set(BOOST_LIBS
        threading
    )
    add_executable_boost_ompi(${TARGET} "${SOURCES}" )

    ##
        
    

endif()
