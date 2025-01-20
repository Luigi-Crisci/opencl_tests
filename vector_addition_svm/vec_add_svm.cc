#include <iostream>
#include <CL/opencl.hpp>

const char* kernelSource = R"(
__kernel void vectorAdd(__global float* a,
                       __global float* b,
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

        // Create and build program
        cl::Program program(context, kernelSource);
        try {
            program.build({device});
        } catch (const std::exception& e) {
            std::cerr << "Build error: " 
                      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) 
                      << std::endl;
            throw e;
        }

        // Create kernel
        cl::Kernel kernel(program, "vectorAdd");

        // Allocate SVM buffers
        float* a = static_cast<float*>(clSVMAlloc(
            context(), 
            CL_MEM_READ_ONLY,
            VECTOR_SIZE * sizeof(float),
            0));
        
        float* b = static_cast<float*>(clSVMAlloc(
            context(),
            CL_MEM_READ_ONLY,
            VECTOR_SIZE * sizeof(float),
            0));
        
        float* c = static_cast<float*>(clSVMAlloc(
            context(),
            CL_MEM_WRITE_ONLY,
            VECTOR_SIZE * sizeof(float),
            0));

        if (!a || !b || !c) {
            throw std::runtime_error("Failed to allocate SVM buffers");
        }

        // Initialize input data
        for (size_t i = 0; i < VECTOR_SIZE; i++) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i);
        }

        // Map SVM buffers
        queue.enqueueMapSVM(a, CL_TRUE, CL_MAP_WRITE, VECTOR_SIZE * sizeof(float));
        queue.enqueueMapSVM(b, CL_TRUE, CL_MAP_WRITE, VECTOR_SIZE * sizeof(float));
        queue.enqueueMapSVM(c, CL_TRUE, CL_MAP_WRITE, VECTOR_SIZE * sizeof(float));

        // Set kernel arguments using SVM pointers
        kernel.setArg(0, a);
        kernel.setArg(1, b);
        kernel.setArg(2, c);
        kernel.setArg(3, static_cast<unsigned int>(VECTOR_SIZE));

        // Execute kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VECTOR_SIZE), cl::NullRange);
        queue.finish();

        // Unmap SVM buffers
        queue.enqueueUnmapSVM(a);
        queue.enqueueUnmapSVM(b);
        queue.enqueueUnmapSVM(c);
        queue.finish();

        // Verify results
        bool correct = true;
        for (size_t i = 0; i < VECTOR_SIZE; i++) {
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

        // Free SVM buffers
        clSVMFree(context(), a);
        clSVMFree(context(), b);
        clSVMFree(context(), c);

    } catch (const std::exception& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.what() << ")" << std::endl;
        return 1;
    }

    return 0;
}