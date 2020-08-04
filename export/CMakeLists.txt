cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(predict_demo)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(demonet detection.cpp)
target_link_libraries(demonet ${OpenCV_LIBS} ${TORCH_LIBRARIES} TorchVision::TorchVision)

# set C++14 to compile PyTorch
set_property(TARGET demonet PROPERTY CXX_STANDARD 14)
