project(Cpp_Example_Eigen LANGUAGES C CXX)
ppr_blue("Build EXECUTABLE: ${PROJECT_NAME}...")
add_executable(${PROJECT_NAME} matmul_eigen.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS ON)

# create simple lib

#ppr_blue("Build LIBRARY: ${PROJECT_NAME}_Lib...")
#add_library(${PROJECT_NAME}_Lib simple_lib.cpp simple_lib.hpp)
#config_lib_output(${PROJECT_NAME}_Lib)
#set_target_properties(${PROJECT_NAME}_Lib PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT result)
if (result)
    set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_LIBRARY_PATH}/Eigen/include/eigen3)

config_bin_output(${PROJECT_NAME})





