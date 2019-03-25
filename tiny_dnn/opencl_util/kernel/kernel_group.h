#pragma once

#include "third_party/CLCudaAPI/clpp11.h"
#include "kernel.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tiny_dnn {

    class KernelGroup {        
        
        public:
            KernelGroup() {}
            
            KernelGroup(std::string source_file, CLCudaAPI::Device* device, 
                CLCudaAPI::Context* context, CLCudaAPI::Queue* queue);
            
            Kernel getKernel(std::string kernel_name);

        private:
            CLCudaAPI::Program program;
            CLCudaAPI::Device* device;
            CLCudaAPI::Context* context;
            CLCudaAPI::Queue* queue;

            std::string loadFileContent(std::string filename);
    };

    KernelGroup::KernelGroup(std::string source_file, CLCudaAPI::Device* device, 
        CLCudaAPI::Context* context, CLCudaAPI::Queue* queue) {
        
        this->device = device;
        this->context = context;
        this->queue = queue;

        program = CLCudaAPI::Program(*context, loadFileContent(source_file));        

        std::vector<std::string> compiler_options = std::vector<std::string>{};

        try {
            program.Build(*device, compiler_options);
        } catch (const CLCudaAPI::CLCudaAPIBuildError &e) {            
            if (program.StatusIsCompilationWarningOrError(e.status())) {
                std::cout << "Compiler error(s)/warning(s) found: "
                    << program.GetBuildInfo(*device) << std::endl;
            }
            throw;
        }
    }

    Kernel KernelGroup::getKernel(std::string kernel_name) {
        return Kernel(kernel_name, &program, context, queue);
    }

    std::string KernelGroup::loadFileContent(std::string filename)
    {
        // TODO ughh thats not ok here
        const std::string sourceFolderPath
            = "/home/artibarti/Desktop/szakdoga/tiny-dnn/tiny_dnn/opencl_util/kernel_sources/";

  	    std::string result;
  	    std::ifstream fs((sourceFolderPath + filename).c_str());
  	    
        if (fs.good())
        {
        	std::stringstream is;
            is << fs.rdbuf();
            result = is.str();
  	    } else {
            throw std::runtime_error("Can't find " + filename);
        }

  	    return result;
    }

} // namespace tiny_dnn
