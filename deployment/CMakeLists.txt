cmake_minimum_required(VERSION 3.14)

project(DEMONET_LibTorch_Example VERSION 0.0.1 LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# This due to some headers importing Python.h
find_package(Python3 COMPONENTS Development)

include_directories(${PROJECT_SOURCE_DIR}/include)

FILE(GLOB DETR_SOURCE_FILES src/*.cpp)

add_executable(inference ${DETR_SOURCE_FILES})
target_compile_features(inference PUBLIC cxx_range_for)

target_link_libraries(
  inference
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  TorchVision::TorchVision
  Python3::Python
)

# set C++14 to compile
set_property(TARGET inference PROPERTY CXX_STANDARD 14)
