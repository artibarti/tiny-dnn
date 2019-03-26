#pragma once

#include "tiny_dnn/util/util.h"
#include "tiny_dnn/containers/shape.h"

namespace tiny_dnn {

    template<typename T>
    class Matrix {

        private:
            std::shared_ptr<T> m_data;
            int elementCount;
            Shape2d shape;

            void setShape(int rows, int cols);

        public:
            Matrix<T>() {}
            Matrix(int rows, int cols);
            Matrix(Matrix<T>& m);
            Matrix<T>& operator= (Matrix<T>& m);

            // size access
            int getElementCount();
            int rowCount();
            int colCount();

            // data access
            T* data();
            T* row(unsigned index);
            T& operator[] (const unsigned index);
            const T& operator[] (const unsigned index) const;

            // for compability checking
            bool isMultipliableWith(Matrix<T>& m);
            bool hasSameDimensionWith(Matrix<T>& m);
    };

    template<typename T>
    Matrix<T>::Matrix(int rows, int cols) {
        setShape(rows, cols);
    }

    template<typename T>
    Matrix<T>::Matrix(Matrix<T>& m) {

        setShape(m.rowCount(), m.colCount());
        for (int i = 0; i<m.getElementCount(); i++) {
            m_data[i] = m[i];
        }
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator= (Matrix<T>& m) {

        setShape(m.rowCount(), m.colCount());
        for (int i = 0; i<m.getElementCount(); i++) {
            m_data[i] = m[i];
        }
    }

    template<typename T>
    int Matrix<T>::getElementCount() {
        return elementCount;
    }

    template<typename T>
    int Matrix<T>::rowCount() {
        return shape.x;
    }

    template<typename T>
    int Matrix<T>::colCount() {
        return shape.y;
    }

    template<typename T>
    T* Matrix<T>::data() {
        return m_data.get();
    }

    template<typename T>
    T* Matrix<T>::row(unsigned index) {
        return data.get()[(index) * shape.y];
    }

    template<typename T>
    T& Matrix<T>::operator[] (const unsigned index) {

        if (index < elementCount) {
            return data.get()[index];
        } else {
            throw std::out_of_range("Index out of bounds");
        }
    }

    template<typename T>
    const T& Matrix<T>::operator[] (const unsigned index) const {

        if (index < elementCount) {
            return data.get()[index];
        } else {
            throw std::out_of_range("Index out of bounds");
        }
    }

    template<typename T>
    bool Matrix<T>::isMultipliableWith(Matrix<T>& m) {
        return shape.y == m.rowCount();
    }
    
    template<typename T>
    bool Matrix<T>::hasSameDimensionWith(Matrix<T>& m) {
        return shape.x == m.rowCount() 
            && shape.y == m.colCount();
    }

    template<typename T>
    void Matrix<T>::setShape(int rows, int cols) {
        
        elementCount = rows * cols;
        shape = Shape2d(rows, cols);
        
        data = std::shared_ptr<T>(
            new T[elementCount], [&](T* ptr){ delete[] ptr; });
    }
}