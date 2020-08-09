cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(demonet)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${PROJECT_SOURCE_DIR}/csrc)

FILE( GLOB source_src
  csrc/backbone.cpp
  csrc/IntermediateLayerGetter.cpp
  csrc/SegmentationModel.cpp
  csrc/DeeplabV3.cpp
  scrc/predict.cpp
)

add_executable(demonet ${source_src})
target_link_libraries(demonet ${OpenCV_LIBS} ${TORCH_LIBRARIES} TorchVision::TorchVision)

# set C++14 to compile PyTorch
set_property(TARGET demonet PROPERTY CXX_STANDARD 14)
