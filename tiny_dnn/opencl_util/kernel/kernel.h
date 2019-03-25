#pragma once

#include "third_party/CLCudaAPI/clpp11.h"
#include "kernel_buffer.h"

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace tiny_dnn {

    class Kernel {

        public:
            Kernel() {}

            Kernel(std::string kernel_name, CLCudaAPI::Program& program, 
                CLCudaAPI::Context* context, CLCudaAPI::Queue* queue);

            void launch(std::vector<size_t> global, std::vector<size_t> local);

            // TODO (should accept vector)
            template<typename T>
            void setArgument(int index,  CLCudaAPI::Buffer<T>& data);

            template<typename T>
            void setArgument(int index,  T t);

            template<typename T>
            void setArgument(int index, Buffer<T> buffer);

            CLCudaAPI::Context& getContext();
            CLCudaAPI::Queue& getQueue();

            

        private:
            CLCudaAPI::Kernel kernel;            
            CLCudaAPI::Context* context;
            CLCudaAPI::Queue* queue;
            CLCudaAPI::Event event;
    };

    Kernel::Kernel(std::string kernel_name, CLCudaAPI::Program& program, 
        CLCudaAPI::Context* context, CLCudaAPI::Queue* queue) {

        this -> context = context;
        this -> queue = queue;        
        kernel = CLCudaAPI::Kernel(program, kernel_name);
        event = CLCudaAPI::Event();
    }

    void Kernel::launch(std::vector<size_t> global, std::vector<size_t> local) {
        this -> kernel.Launch(*queue, global, local, event.pointer());
        queue->Finish(event);
    }

    template<typename T>
    void Kernel::setArgument(int index,  T t) {
        kernel.SetArgument(index, t);
    }

    template<typename T>
    void Kernel::setArgument(int index, Buffer<T> buffer) {
        kernel.SetArgument(index, buffer.data());
    }

    CLCudaAPI::Context& Kernel::getContext() {
        return *context;
    }
    
    CLCudaAPI::Queue& Kernel::getQueue() {
        return *queue;
    }

} // namespace tiny_dnn
