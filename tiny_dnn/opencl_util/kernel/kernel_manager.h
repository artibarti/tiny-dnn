#pragma once

#include "third_party/CLCudaAPI/clpp11.h"
#include "kernel_group.h"

#include <stdexcept>
#include <string>
#include <map>

namespace tiny_dnn {

    class KernelManager {

        public:
            
            static KernelManager& getInstance() {
                static KernelManager instance;
                return instance;
            }

            KernelGroup getKernelGroup(std::string source_file);

        private:
            
            size_t platform_id;
            size_t device_id;
            
            CLCudaAPI::Platform platform;
            CLCudaAPI::Device device;
            CLCudaAPI::Context context;
            CLCudaAPI::Queue queue;

            std::map<std::string, KernelGroup> cachedKernelGroups;
            
            KernelManager() {
                
                platform_id = size_t{0};
                device_id = size_t{0};
        
                platform = CLCudaAPI::Platform(platform_id);
                device   = CLCudaAPI::Device(platform, device_id);
                context  = CLCudaAPI::Context(device);
                queue    = CLCudaAPI::Queue(context, device);
            }
    };

    KernelGroup KernelManager::getKernelGroup(std::string source_file) {
        
        if (cachedKernelGroups.find(source_file) != cachedKernelGroups.end()) {
            return cachedKernelGroups[source_file];                        
        } else {
            KernelGroup kernelGroup = KernelGroup(source_file, device, context, queue);
            cachedKernelGroups[source_file] = kernelGroup;
            return kernelGroup;
        }
    }

} // namespace tiny_dnn