#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

int main(){
    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue queue;

    cl::Platform::get(&platform);
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    context = cl::Context(devices[0]);
    queue = cl::CommandQueue(context, devices[0]);

    std::cout << "Running on device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

    std::cout << "CL_DEVICE_IL_VERSION: " << devices[0].getInfo<CL_DEVICE_IL_VERSION>() << std::endl;
}