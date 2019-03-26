#pragma once

#ifdef USE_OPENCL
  #include "third_party/CLCudaAPI/clpp11.h"
#endif

inline void printAvailableDevices(const size_t platform_id, const size_t device_id) {

  #ifdef USE_OPENCL

    auto platform = CLCudaAPI::Platform(platform_id);
    auto device   = CLCudaAPI::Device(platform, device_id);

    printf("\n## Printing device information...\n");
    printf(" > Platform ID                  %zu\n", platform_id);
    printf(" > Device ID                    %zu\n", device_id);
    printf(" > Framework version            %s\n", device.Version().c_str());
    printf(" > Vendor                       %s\n", device.Vendor().c_str());
    printf(" > Device name                  %s\n", device.Name().c_str());
    printf(" > Device type                  %s\n", device.Type().c_str());
    printf(" > Max work-group size          %zu\n", device.MaxWorkGroupSize());
    printf(" > Max thread dimensions        %zu\n",device.MaxWorkItemDimensions());
    printf(" > Max work-group sizes:\n");

    for (auto i = size_t{0}; i < device.MaxWorkItemDimensions(); ++i) {
        printf("   - in the %zu-dimension         %zu\n", i, device.MaxWorkItemSizes()[i]);
    }
  
    printf(" > Local memory per work-group  %zu bytes\n", device.LocalMemSize());
    printf(" > Device capabilities          %s\n", device.Capabilities().c_str());
    printf(" > Core clock rate              %zu MHz\n", device.CoreClock());
    printf(" > Number of compute units      %zu\n", device.ComputeUnits());
    printf(" > Total memory size            %zu bytes\n", device.MemorySize());
    printf(" > Maximum allocatable memory   %zu bytes\n", device.MaxAllocSize());
    printf(" > Memory clock rate            %zu MHz\n", device.MemoryClock());
    printf(" > Memory bus width             %zu bits\n", device.MemoryBusWidth());

  #else
    nn_warn("TinyDNN was not build with OpenCL or CUDA support.");
  #endif

}
