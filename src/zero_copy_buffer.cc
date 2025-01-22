#include <CL/opencl.hpp>
#include <iostream>

int main() {
  // Get platform and device
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    throw std::runtime_error("No OpenCL platforms found");
  }

  cl::Platform platform = platforms[0];
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if (devices.empty()) {
    throw std::runtime_error("No OpenCL devices found");
  }
  cl::Device device = devices[0];

  // Check if device supports SVM
  cl_device_svm_capabilities svm_caps =
      device.getInfo<CL_DEVICE_SVM_CAPABILITIES>();

  if (!(svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
    throw std::runtime_error("Device does not support coarse-grained SVM");
  }

  // Create context and queue
  cl::Context context(device);
  cl::CommandQueue queue(context, device);


  int data[8];

  cl_int err;
  cl_mem res = clImportMemoryARM(context(),CL_MEM_READ_WRITE, nullptr , (void*) data, 8 * sizeof(int),&err);
    
  


}