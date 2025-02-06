#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>


auto init_opencl(){
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

        return queue;
}


int main(){

    auto queue = init_opencl();

    std::ifstream file("/home/orangepi/lcrisci/opencl_tests/src/hello_world.spv", std::ios::in);
    std::vector<char> ir(std::istreambuf_iterator<char>(file), {});
    file.close();

    for(auto& c : ir){
        std::cout << c;
    }


    cl_int err;
    cl::Program program(ir, false, &err);
    
    auto handle_opencl_err = [&](cl_int res){
        if (res!= CL_SUCCESS){
            auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
            for ( auto [dev, log] : log) {
                std::cout << "Build log for device: " << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
                std::cout << log << std::endl;
            }
        }
    };
    
    handle_opencl_err(err);

    auto res = program.build("-cl-std=CL2.0");
    handle_opencl_err(res);

    return 0;
}