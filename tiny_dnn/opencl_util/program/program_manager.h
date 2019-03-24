#pragma once

#include "third_party/CLCudaAPI/clpp11.h"
#include "program.h"
#include <string>
#include <map>
#include <stdexcept>

namespace tiny_dnn {

    class ProgramManager {

        public:
            
            static ProgramManager& getInstance() {
                static ProgramManager instance;
                return instance;
            }
            
            CLProgram getProgram(std::string source_file);
        
        private:
            
            size_t platform_id;
            size_t device_id;
            
            CLCudaAPI::Platform platform;
            CLCudaAPI::Device device;
            CLCudaAPI::Context context;
            CLCudaAPI::Queue queue;

            std::map<std::string, CLProgram> compiledPrograms;
            
            ProgramManager() {
                
                platform_id = size_t{0};
                device_id = size_t{0};
        
                platform = CLCudaAPI::Platform(platform_id);
                device   = CLCudaAPI::Device(platform, device_id);
                context  = CLCudaAPI::Context(device);
                queue    = CLCudaAPI::Queue(context, device);
            }
    };

    CLProgram ProgramManager::getProgram(std::string source_file) {
        
        if (compiledPrograms.find(source_file) != compiledPrograms.end()) {
            return compiledPrograms[source_file];                        
        } else {
            CLProgram program = CLProgram(&context, &device, &queue, source_file);
            compiledPrograms[source_file] = program;
            return program;
        }
    }

} // namespace tiny_dnn