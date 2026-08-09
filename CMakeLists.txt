# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(deploy_pico2/pico_sdk_import.cmake)

project(app C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1

add_executable(app
    deploy_pico2/main.c
  codegen/Source/genModel.c
  codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW_fpreq.c
  codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW_fpreq.c
  codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW_fpreq.c
  codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW_fpreq.c
  TinyEngine/src/kernels/fp_requantize_op/add_fpreq.c
  TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_fpreq.c
  TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch16_fpreq.c
  TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch24_fpreq.c
  TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch48_fpreq.c
  TinyEngine/src/kernels/fp_requantize_op/convolve_s8_kernel3_inputch3_stride2_pad1_fpreq.c
  TinyEngine/src/kernels/fp_requantize_op/mat_mul_kernels_fpreq.c
  TinyEngine/src/kernels/int_forward_op/avgpooling.c
    )


target_include_directories(app PRIVATE
  deploy/cmsis_shim
  TinyEngine/include
  codegen/Include
  TinyEngine/third_party/CMSIS/CMSIS/Core/Include
  TinyEngine/third_party/CMSIS/CMSIS/DSP/Include
  TinyEngine/third_party/CMSIS/CMSIS/NN/Include
)

# pull in common dependencies
target_link_libraries(app pico_stdlib m cmsis_core)
target_compile_options(app PRIVATE 
-Wno-error=incompatible-pointer-types
-Wno-error=implicit-function-declaration
-Wno-error=int-conversion

)
# create map/bin/hex file etc.
pico_add_extra_outputs(app)
# ... existing cmake configuration ...
# pico_set_binary_type(app no_flash) # Directs linker to RAM
# Enable printing over USB CDC (Virtual COM Port)
pico_enable_stdio_usb(app 1)
# Generate a .su file for every source file compiled
target_compile_options(app PRIVATE "-fstack-usage")

# add url via pico_set_program_url
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--print-memory-usage")