// One-stop header.
#include <torch/script.h>

// header for torchvision
#include <torchvision/nms.h>

// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_SIZE 320
#define kCHANNELS 3
#define kTOP_K 3

static auto registry = torch::RegisterOperators().op("torchvision::nms", &nms);

bool LoadImage(std::string file_name, cv::Mat &image,
               float &width, float &height) {
  image = cv::imread(file_name);  // CV_8UC3
  if (image.empty() || !image.data) {
    return false;
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  width = float(image.cols);
  height = float(image.rows);
  std::cout << "== image size: " << image.size() << " ==" << std::endl;

  // scale image to fit
  cv::Size scale(kIMAGE_SIZE, kIMAGE_SIZE);
  cv::resize(image, image, scale);
  std::cout << "== simply resize: " << image.size() << " ==" << std::endl;

  // convert [unsigned int] to [float]
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

  return true;
}

bool LoadImageNetLabel(std::string file_name,
                       std::vector<std::string> &labels) {
  std::ifstream ifs(file_name);
  if (!ifs) {
    return false;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    labels.push_back(line);
  }
  return true;
}

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: detect <path-to-exported-script-module> "
                 "<path-to-lable-file>"
              << std::endl;
    return -1;
  }

  torch::jit::script::Module module = torch::jit::load(argv[1]);
  std::cout << "== Switch to GPU mode" << std::endl;
  // to GPU
  module.to(torch::kCUDA);

  std::cout << "== DEMONET loaded!\n";
  std::vector<std::string> labels;
  if (LoadImageNetLabel(argv[2], labels)) {
    std::cout << "== Label loaded! Let's try it\n";
  } else {
    std::cerr << "Please check your label file path." << std::endl;
    return -1;
  }

  std::string file_name = "";
  cv::Mat image;
  float img_width = 0.;
  float img_height = 0.;
  while (true) {
    std::cout << "== Input image path: [enter Q to exit]" << std::endl;
    std::cin >> file_name;
    if (file_name == "Q") {
      break;
    }
    if (LoadImage(file_name, image, img_width, img_height)) {
      auto input_tensor = torch::from_blob(
          image.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
      input_tensor = input_tensor.permute({0, 3, 1, 2});
      input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
      input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
      input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

      float image_size[] = {img_height, img_width};
      torch::Tensor image_shapes = torch::from_blob(image_size, {1, 2});

      // to GPU
      input_tensor = input_tensor.to(torch::kCUDA);
      image_shapes = image_shapes.to(torch::kCUDA);

      // std::cout << "tensors: " << input_tensor << std::endl;
      // std::cout << "size: " << image_shapes << std::endl;

      auto output = module.forward({input_tensor, image_shapes});

      std::cout << "output: " << output << std::endl;

      auto out1 = output.toTuple();
      auto dets = out1->elements().at(1).toGenericDict();

      std::cout << "dets: " << dets << std::endl;
      // auto det0 = dets.get(0).toGenericDict() ;
      // at::Tensor scores = det0.at("scores").toTensor();
      // at::Tensor labels = det0.at("labels").toTensor();
      // at::Tensor boxes = det0.at("boxes").toTensor();

      // std::cout << "scores: " << scores << std::endl;
      // std::cout << "lables: " << labels << std::endl;
      // std::cout << "boxes: " << boxes << std::endl;

    } else {
      std::cout << "Can't load the image, please check your path." << std::endl;
    }
  }
  return 0;
}
