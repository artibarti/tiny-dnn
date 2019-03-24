#pragma once

#include <string>
#include <vector>
#include "third_party/CLCudaAPI/clpp11.h"
#include <iostream>
#include <stdexcept>
#include "tiny_dnn/config.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

    class CLKernel {

        public:
            CLKernel() {}

            CLKernel(CLCudaAPI::Program* program, CLCudaAPI::Context* context,
                CLCudaAPI::Queue* queue, std::string kernel_name) {
                
                kernel = CLCudaAPI::Kernel(*program, kernel_name);
                this -> context = context;
                this -> queue = queue;
                event = CLCudaAPI::Event();
            }

            template<typename T>
            void setArgument(int index,  CLCudaAPI::Buffer<T>* data) {
                kernel.SetArgument(index, *data);
            }

            void launch(std::vector<size_t> global = {512, 1, 1}, 
                std::vector<size_t> local = {256, 1, 1}) {

                kernel.Launch(*queue, global, local, event.pointer());
                queue->Finish(event);
            }

            CLCudaAPI::Context* getContext() {
                return context;
            }

            CLCudaAPI::Queue* getQueue() {
                return queue;
            }

        private:
            CLCudaAPI::Kernel kernel;            
            CLCudaAPI::Context* context;
            CLCudaAPI::Queue* queue;
            CLCudaAPI::Event event;
            std::vector<CLCudaAPI::Buffer<float>> params;
    };

} // namespace tiny_dnn
