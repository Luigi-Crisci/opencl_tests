#include <iostream>
#include <CL/opencl.hpp>

const char* kernelSource = R"(
__kernel void vectorAdd(__global const float* a,
                       __global const float* b,
                       __global float* c,
                       const unsigned int n)
{
    int id = get_global_id(0);
    if (id < n)
        c[id] = a[id] + b[id];
}
)";

#define VECTOR_SIZE 1024

int main() {
    try {
        // Get available platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        // Select the first platform
        cl::Platform platform = platforms[0];
        std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        // Get available devices
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found");
        }

        // Select the first device
        cl::Device device = devices[0];
        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Create OpenCL context
        cl::Context context(device);

        // Create command queue
        cl::CommandQueue queue(context, device);

        // Create program
        cl::Program program(context, kernelSource);
        
        // Build program
        try {
            program.build({device});
        } catch (const std::exception e) {
            std::cerr << "Build error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            throw e;
        }

        // Create kernel
        cl::Kernel kernel(program, "vectorAdd");

        // Prepare input data
        std::vector<float> a(VECTOR_SIZE);
        std::vector<float> b(VECTOR_SIZE);
        std::vector<float> c(VECTOR_SIZE);

        // Initialize input vectors
        for (int i = 0; i < VECTOR_SIZE; i++) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i);
        }

        // Create buffers
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * VECTOR_SIZE, a.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * VECTOR_SIZE, b.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY,
                          sizeof(float) * VECTOR_SIZE);

        // Set kernel arguments
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, static_cast<unsigned int>(VECTOR_SIZE));

        // Execute kernel
        cl::NDRange global(VECTOR_SIZE);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
        queue.finish();

        // Read results
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0,
                              sizeof(float) * VECTOR_SIZE, c.data());

        // Verify results
        bool correct = true;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (std::abs(c[i] - (a[i] + b[i])) > 1e-5) {
                std::cout << "Verification failed at index " << i << std::endl;
                std::cout << "Expected: " << a[i] + b[i] << ", Got: " << c[i] << std::endl;
                correct = false;
                break;
            }
        }

        if (correct) {
            std::cout << "Vector addition completed successfully!" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.what() << ")" << std::endl;
        return 1;
    }

    return 0;
}