#pragma once

#include <string>
#include <vector>
#include "third_party/CLCudaAPI/clpp11.h"

namespace tiny_dnn {

    class CLProgram {        
        
        public:
            CLProgram() {}
            CLProgram(CLCudaAPI::Context context, CLCudaAPI::Device device, std::string source_file);
            
            CLCudaAPI::Kernel getKerel(std::string kernel_name);
            std::string getSourceFile();

        private:
            std::string source_file;
            CLCudaAPI::Program program;
            std::vector<std::string> compiler_options = std::vector<std::string>{};

            std::string loadFileContent(std::string filename);
    };

    CLProgram::CLProgram(CLCudaAPI::Context context, CLCudaAPI::Device device, std::string source_file) {
        
        source_file = source_file;

        program = CLCudaAPI::Program(context, loadFileContent(source_file));
        
        program.Build(device, compiler_options);        
        
        /*
        try {
            program.Build(device, compiler_options);
        } catch (const CLCudaAPI::CLCudaAPIBuildError e) {
            if (program.StatusIsCompilationWarningOrError(e.status())) {
                std::cout << program.GetBuildInfo(device);
            }            
            throw;
        }
        */
    }

    CLCudaAPI::Kernel CLProgram::getKerel(std::string kernel_name) {
        return CLCudaAPI::Kernel(program, kernel_name);
    }

    std::string CLProgram::getSourceFile() {
        return source_file;
    }

    std::string CLProgram::loadFileContent(std::string filename)
    {
  	    std::string result;
  	    std::ifstream fs(filename.c_str());
  	    
        if (fs.good())
        {
        	std::stringstream is;
            is << fs.rdbuf();
            result = is.str();
  	    }

  	    return result;
    }
}