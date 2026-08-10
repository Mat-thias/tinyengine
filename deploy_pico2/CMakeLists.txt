# ----------------------------------------------------------------------
# Project: Right In-Place Convolution
# Title:   CMakeLists.txt
#
# Reference papers:
#    Yet to be published
# Contact authors:
#  - Opegbemi Matthias Busoye, matthias@powerlabstech.com
#  - Tolulope Matthew Busoye, matthew@powerlabstech.com
#  - Eghonghon-aye Eigbe, eghonghon@powerlabstech.com
#
# Target ISA:  ARMv8-M (Cortex-M33, RP2350)
# ----------------------------------------------------------------------
#
# Builds the generated TinyEngine model for the Raspberry Pi Pico 2:
#
#     cmake -S deploy_pico2 -B deploy_pico2/build -DPICO_BOARD=pico2
#     cmake --build deploy_pico2/build
#
# Sources come from codegen/, which holds whichever model was generated last
# by examples/mcunet_inx.py, so no model is named here.

# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
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
include(pico_sdk_import.cmake)

project(blink C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

get_filename_component(TINYENGINE_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)

file(GLOB TINYENGINE_GENERATED_KERNELS CONFIGURE_DEPENDS
  ${TINYENGINE_ROOT}/codegen/Source/depthwise_*_inplace_CHW_fpreq.c)

add_executable(blink
  main.c
  ${TINYENGINE_ROOT}/codegen/Source/genModel.c
  ${TINYENGINE_GENERATED_KERNELS}
  ${TINYENGINE_ROOT}/TinyEngine/src/kernels/fp_requantize_op/add_fpreq.c
  ${TINYENGINE_ROOT}/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_fpreq.c
  ${TINYENGINE_ROOT}/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch8_fpreq.c
  ${TINYENGINE_ROOT}/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch16_fpreq.c
  ${TINYENGINE_ROOT}/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch24_fpreq.c
  ${TINYENGINE_ROOT}/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch48_fpreq.c
  ${TINYENGINE_ROOT}/TinyEngine/src/kernels/fp_requantize_op/convolve_s8_kernel3_inputch3_stride2_pad1_fpreq.c
  ${TINYENGINE_ROOT}/TinyEngine/src/kernels/fp_requantize_op/mat_mul_kernels_fpreq.c
  ${TINYENGINE_ROOT}/TinyEngine/src/kernels/int_forward_op/avgpooling.c
)


target_include_directories(blink PRIVATE
  ${TINYENGINE_ROOT}/TinyEngine/include
  ${TINYENGINE_ROOT}/codegen/Include
  ${TINYENGINE_ROOT}/TinyEngine/third_party/CMSIS/CMSIS/Core/Include
  ${TINYENGINE_ROOT}/TinyEngine/third_party/CMSIS/CMSIS/DSP/Include
  ${TINYENGINE_ROOT}/TinyEngine/third_party/CMSIS/CMSIS/NN/Include
)


target_link_libraries(blink pico_stdlib m cmsis_core)

pico_add_extra_outputs(blink)

pico_enable_stdio_usb(blink 1)
# add url via pico_set_program_url
