#pragma once

#include <string>
#include <vector>
#include "third_party/CLCudaAPI/clpp11.h"
#include "kernel_function.h"
#include <iostream>
#include <stdexcept>

namespace tiny_dnn {

    class CLProgram {        
        
        public:
            CLProgram() {}
            
            CLProgram(CLCudaAPI::Context* context, CLCudaAPI::Device* device, 
                CLCudaAPI::Queue* queue, std::string source_file);
            
            CLKernel getKernel(std::string kernel_name);
            std::string getSourceFile();

        private:
            CLCudaAPI::Program program;
            CLCudaAPI::Context* context;
            CLCudaAPI::Device* device;
            CLCudaAPI::Queue* queue;

            std::string source_file;            
            std::vector<std::string> compiler_options = std::vector<std::string>{};
            std::string loadFileContent(std::string filename);
    };

    CLProgram::CLProgram(CLCudaAPI::Context* context, CLCudaAPI::Device* device,
        CLCudaAPI::Queue* queue, std::string source_file) {
        
        this->device = device;
        this->queue = queue;
        this->context = context;
        this->source_file = source_file;

        program = CLCudaAPI::Program(*context, loadFileContent(source_file));        
        
        if (program.Build(*device, compiler_options) != CLCudaAPI::BuildStatus::kSuccess) {
            throw std::runtime_error("Error while compiling kernel: " + source_file);
        }
    }

    CLKernel CLProgram::getKernel(std::string kernel_name) {
        return CLKernel(&program, context, queue, kernel_name);
    }

    std::string CLProgram::getSourceFile() {
        return source_file;
    }

    std::string CLProgram::loadFileContent(std::string filename)
    {
        const std::string sourceFolderPath
            = "/home/artibarti/Desktop/szakdoga/tiny-dnn/tiny_dnn/opencl_util/kernels/";

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
