; ModuleID = 'cpp_opencl.cpp'
source_filename = "cpp_opencl.cpp"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

$_Z3addIfET_S0_S0_ = comdat any

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_kernel void @test(float addrspace(1)* noundef %0, float addrspace(1)* noundef %1) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  %3 = alloca float addrspace(1)*, align 4
  %4 = alloca float addrspace(1)*, align 4
  %5 = alloca i32, align 4
  %6 = alloca float, align 4
  %7 = alloca float, align 4
  store float addrspace(1)* %0, float addrspace(1)** %3, align 4
  store float addrspace(1)* %1, float addrspace(1)** %4, align 4
  %8 = call spir_func noundef i32 @_Z13get_global_idj(i32 noundef 0) #3
  store i32 %8, i32* %5, align 4
  %9 = load float addrspace(1)*, float addrspace(1)** %4, align 4
  %10 = load i32, i32* %5, align 4
  %11 = getelementptr inbounds float, float addrspace(1)* %9, i32 %10
  %12 = load float, float addrspace(1)* %11, align 4
  store float %12, float* %6, align 4
  %13 = load float addrspace(1)*, float addrspace(1)** %4, align 4
  %14 = load i32, i32* %5, align 4
  %15 = add i32 %14, 1
  %16 = getelementptr inbounds float, float addrspace(1)* %13, i32 %15
  %17 = load float, float addrspace(1)* %16, align 4
  store float %17, float* %7, align 4
  %18 = load float, float* %6, align 4
  %19 = load float, float* %7, align 4
  %20 = call spir_func noundef float @_Z3addIfET_S0_S0_(float noundef %18, float noundef %19) #4
  %21 = load float addrspace(1)*, float addrspace(1)** %3, align 4
  %22 = load i32, i32* %5, align 4
  %23 = getelementptr inbounds float, float addrspace(1)* %21, i32 %22
  store float %20, float addrspace(1)* %23, align 4
  ret void
}

; Function Attrs: convergent nounwind readnone willreturn
declare dso_local spir_func noundef i32 @_Z13get_global_idj(i32 noundef) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func noundef float @_Z3addIfET_S0_S0_(float noundef %0, float noundef %1) #2 comdat {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = fadd float %5, %6
  ret float %7
}

attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind readnone willreturn "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind readnone willreturn }
attributes #4 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 2, i32 0}
!3 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!4 = !{i32 1, i32 1}
!5 = !{!"none", !"none"}
!6 = !{!"float*", !"float*"}
!7 = !{!"", !""}
