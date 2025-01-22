MACRO (ADD_SAMPLE name)
  ADD_EXECUTABLE(${name} ${name}.cc)
  # add_compile_options("--verbose")
  TARGET_LINK_LIBRARIES(${name} OpenCL::OpenCL /home/orangepi/lcrisci/OpenCL-SDK/install/lib/libOpenCLExt.a)
  # Add opencl include files 
  TARGET_INCLUDE_DIRECTORIES(${name} PRIVATE ${OpenCL_INCLUDE_DIRS})
endmacro(ADD_SAMPLE)

ADD_SAMPLE(vec_add_svm)
ADD_SAMPLE(get_platforms)
ADD_SAMPLE(vec_add_buffer)
ADD_SAMPLE(svm_memops)
ADD_SAMPLE(zero_copy_buffer)
