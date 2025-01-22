#include <CL/opencl.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#include <cassert>

constexpr int vector_size = 1024;

const char *kernel_iota = R"(
__kernel void iota(__global int* a,
                       const int n)
{
    int id = get_global_id(0);
    if (id < n)
        a[id] = id;
}
)";

bool verify(int* b, const int size)
{
    for (int i = 0; i < size; i++)
    {
        if (b[i] != i)
        {
            std::cout << "Verification failed at index " << i << std::endl;
            std::cout << "Expected: " << i << ", Got: " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void reset(int *a, int *b)
{
    for (int i = 0; i < vector_size; i++)
    {
        a[i] = 0;
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

    cl::Platform platform;
    std::vector<cl::Device> devices;
    for (auto &p : platforms)
    {
        platform = p;
        p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (!devices.empty()) break;
    }
    if (devices.empty())
    {
        throw std::runtime_error("No OpenCL devices found");
    }

    cl::Device device = devices[0];
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>()
              << std::endl;
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

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
    // Blocking
    // std::cout << "Blocking" << std::endl;
    // std::iota(a, a + vector_size, 0);
    // queue.enqueueMemcpySVM(b, a, true, vector_size * sizeof(int));
    // std::cout << "Verification "
    //           << (verify(b, vector_size) ? "PASSED" : "FAILED") << std::endl;
    // // Non-blocking
    // std::cout << "Non-blocking" << std::endl;
    // reset(a, b);
    // std::iota(a, a + vector_size, 0);
    // cl::Event event;
    // queue.enqueueMemcpySVM(b, a, false, vector_size * sizeof(int), nullptr,
    //                        &event);
    // event.wait();
    // std::cout << "Verification "
    //           << (verify(b, vector_size) ? "PASSED" : "FAILED") << std::endl;

    // Non-blocking, init with kernel
    std::cout << "Non-blocking, init with kernel" << std::endl;
    reset(a, b);
    for(int i = 0; i < vector_size; i++){
        std::cout << a[i];
    }
    cl::Program program(context, kernel_iota);
    program.build({ device });
    cl::Kernel kernel(program, "iota");

    // queue.enqueueMapSVM(a, CL_TRUE, CL_MAP_WRITE, vector_size * sizeof(int));
    kernel.setArg(0, a);
    kernel.setArg(1, vector_size);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vector_size),
                               cl::NullRange);
    // queue.enqueueUnmapSVM(a);
    queue.finish();

    // queue.enqueueMemcpySVM(b, a, true, vector_size * sizeof(int));
    std::cout << "Verification "
              << (verify(a, vector_size) ? "PASSED" : "FAILED") << std::endl;
}