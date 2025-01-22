#include <CL/opencl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

constexpr int vector_size = 1024;

const char* kernel_iota = R"(
__kernel void iota(__global float* a,
                       const unsigned int n)
{
    int id = get_global_id(0);
    if (id < n)
        a[id] = n;
}
)";

bool verify(int* a, int *b, const int size){
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            std::cout << "Verification failed at index " << i << std::endl;
            std::cout << "Expected: " << a[i] << ", Got: " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void reset(int* a, int* b){
    for (int i = 0; i < vector_size; i++) {
        a[i] = i;
        b[i] = 0;
    }
}

int main()
{
    // Get platform and device
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
    {
        throw std::runtime_error("No OpenCL platforms found");
    }

    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty())
    {
        throw std::runtime_error("No OpenCL devices found");
    }
    cl::Device device = devices[0];

    // Check if device supports SVM
    cl_device_svm_capabilities svm_caps =
        device.getInfo<CL_DEVICE_SVM_CAPABILITIES>();

    if (!(svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER))
    {
        throw std::runtime_error("Device does not support coarse-grained SVM");
    }

    // Create context and queue
    cl::Context context(device);
    cl::CommandQueue queue(context, device);


    // Allocate two SVM buffers
    int *a = (int *)clSVMAlloc(context(), CL_MEM_READ_WRITE,
                                   vector_size * sizeof(int), 0);
    int *b = (int *)clSVMAlloc(context(), CL_MEM_READ_WRITE,
                                   vector_size * sizeof(int), 0);

    if (!a || !b)
    {
        throw std::runtime_error("Failed to allocate SVM buffers");
    }

    reset(a, b);
    //Blocking 
    std::cout << "Blocking" << std::endl;
    queue.enqueueMemcpySVM(b, a, true, vector_size * sizeof(int));
    std::cout << "Verification " << (verify(a, b, vector_size) ? "PASSED" : "FAILED") << std::endl;
    // Non-blocking
    std::cout << "Non-blocking" << std::endl;
    reset(a, b);
    cl::Event event;
    queue.enqueueMemcpySVM(b, a, false, vector_size * sizeof(int), nullptr, &event);
    event.wait();
    std::cout << "Verification " << (verify(a, b, vector_size) ? "PASSED" : "FAILED") << std::endl;
    
    //Non-blocking, init with kernel
    std::cout << "Non-blocking, init with kernel" << std::endl;
    reset(a, b);
    cl::Program program(context, kernel_iota);
    program.build({device});
    cl::Kernel kernel(program, "iota");

    queue.enqueueMapSVM(a, CL_TRUE, CL_MAP_WRITE, vector_size * sizeof(int));
    kernel.setArg(0, a);
    kernel.setArg(1, vector_size);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vector_size), cl::NullRange, nullptr, &event);
    event.wait();
    queue.enqueueUnmapSVM(a);
    queue.enqueueMemcpySVM(b, a, true, vector_size * sizeof(int));
    std::cout << "Verification " << (verify(a, b, vector_size) ? "PASSED" : "FAILED") << std::endl;

}