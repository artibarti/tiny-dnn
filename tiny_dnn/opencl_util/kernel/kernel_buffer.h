#pragma once

#include "third_party/CLCudaAPI/clpp11.h"
#include <vector>

namespace tiny_dnn {

    template<typename T>
    class Buffer {

        public:
            Buffer() {}
            Buffer(CLCudaAPI::Context& context, CLCudaAPI::Queue& queue, size_t size);
            
            void write(const std::vector<T>& data, size_t size, int offset = 0);
            void write(const T* data, size_t size, int offset = 0);
            
            void read(std::vector<T>& result, size_t size, int offset = 0);
            void read(T* result, size_t size, int offset);

            CLCudaAPI::Buffer<T>& data();

        private:
          CLCudaAPI::Buffer<T>* buffer;
          CLCudaAPI::Context* context;
          CLCudaAPI::Queue* queue;
    };

    template<typename T>
    Buffer<T>::Buffer(CLCudaAPI::Context& context, CLCudaAPI::Queue& queue, size_t size) {
        
        this->context = &context;
        this->queue = &queue;        
        buffer = new CLCudaAPI::Buffer<T>(context, size);
    }    

    template<typename T>
    void Buffer<T>::write(const std::vector<T>& data, size_t size, int offset) {
        buffer -> Write(*queue, size, data, offset);
    }

    template<typename T>
    void Buffer<T>::write(const T* data, size_t size, int offset) {
        buffer -> Write(*queue, size, data, offset);
    }

    template<typename T>    
    void Buffer<T>::read(std::vector<T>& result, size_t size, int offset) {
        buffer -> Read(*queue, size, result, offset);
    }

    template<typename T>    
    void Buffer<T>::read(T* result, size_t size, int offset) {
        buffer -> Read(*queue, size, result, offset);
    }

    template<typename T>    
    CLCudaAPI::Buffer<T>& Buffer<T>::data() {
        return *buffer;
    }
}