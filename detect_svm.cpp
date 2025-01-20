#include <CL/cl.hpp>


int main(){

cl_device_svm_capabilities caps;

cl_int err = clGetDeviceInfo(
    deviceID,
    CL_DEVICE_SVM_CAPABILITIES,
    sizeof(cl_device_svm_capabilities),
    &caps,
    0
  );

}
