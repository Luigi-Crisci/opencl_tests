#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>


auto init_opencl(){
    // Get platform and device
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
    {
        throw std::runtime_error("No OpenCL platforms found");
    }

    cl::Platform platform;
    std::vector<cl::Device> devices;
    for (auto &p : platforms)
    {
        platform = p;
        p.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if (!devices.empty()) break;
    }
    if (devices.empty())
    {
        throw std::runtime_error("No OpenCL devices found");
    }

    cl::Device device = devices[0];

    // Create context and queue
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    std::cout << "Running on device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    return std::tuple{queue, context, device};
}


char* kernel = R"(
__kernel void iota(__global float* a,
                       const unsigned int n)
{
    int id = get_global_id(0);
    if (id < n)
        a[id] = n;
}
)";


int main(){

    auto [queue, context, device] = init_opencl();

    std::ifstream file("/home/lcrisci/workspace/librert/opencl_tests/src/iota_kernel.spv", std::ios::in);
    std::vector<char> ir(std::istreambuf_iterator<char>(file), {});
    file.close();

    cl_int err;
    cl::Program program(context, ir, false, &err);
    
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

    auto res = program.build(device,"-cl-std=CL2.0");
    handle_opencl_err(res);

    auto kernel = cl::Kernel(program, "iota", &err);
    handle_opencl_err(err);

    // launch kernel 
    cl::Buffer buffer(context, CL_MEM_READ_WRITE, sizeof(float));
    kernel.setArg(0, buffer);
    kernel.setArg(1, 42);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
    queue.finish();

    float result;
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(float), &result);

    std::cout << "Result: " << result << std::endl;

    return 0;
}