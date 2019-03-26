#pragma once

#include "tiny_dnn/util/util.h"
#include "tiny_dnn/containers/shape.h"

namespace tiny_dnn {

    template<typename T>
    class Matrix {

        private:
            std::shared_ptr<T> m_data;
            unsigned elementCount;
            Shape2d shape;

            void setShape(unsigned rows, unsigned cols);

        public:
            Matrix<T>() {}
            Matrix(unsigned rows, unsigned cols);
            Matrix(Matrix<T>& m);
            Matrix<T>& operator= (Matrix<T>& m);

            // size access
            unsigned getElementCount() const;
            unsigned rowCount() const;
            unsigned colCount() const;

            void resize(unsigned rows, unsigned cols);

            // data access
            T* data();
            T* row(unsigned index);
            T& operator[] (unsigned index);
            const T& operator[] (unsigned index) const;

            // for compability checking
            bool isMultipliableWith(const Matrix<T>& m) const;
            bool hasSameDimensionWith(const Matrix<T>& m) const;
    };

    template<typename T>
    Matrix<T>::Matrix(unsigned rows, unsigned cols) {
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
    unsigned Matrix<T>::getElementCount() const {
        return elementCount;
    }

    template<typename T>
    unsigned Matrix<T>::rowCount() const {
        return shape.x;
    }

    template<typename T>
    unsigned Matrix<T>::colCount()  const {
        return shape.y;
    }

    template<typename T>
    void Matrix<T>::resize(unsigned rows, unsigned cols) {
        if (shape.x != rows || shape.y != cols) {
            setShape(rows, cols);
        }
    }

    template<typename T>
    T* Matrix<T>::data() {
        return m_data.get();
    }

    template<typename T>
    T* Matrix<T>::row(unsigned index) {
        return m_data.get()[(index) * shape.y];
    }

    template<typename T>
    T& Matrix<T>::operator[] (unsigned index) {

        if (index < elementCount) {
            return m_data.get()[index];
        } else {
            throw std::out_of_range("Index out of bounds");
        }
    }

    template<typename T>
    const T& Matrix<T>::operator[] (unsigned index) const {

        if (index < elementCount) {
            return m_data.get()[index];
        } else {
            throw std::out_of_range("Index out of bounds");
        }
    }

    template<typename T>
    bool Matrix<T>::isMultipliableWith(const Matrix<T>& m) const {
        return shape.y == m.rowCount();
    }
    
    template<typename T>
    bool Matrix<T>::hasSameDimensionWith(const Matrix<T>& m) const {
        return shape.x == m.rowCount() 
            && shape.y == m.colCount();
    }

    template<typename T>
    void Matrix<T>::setShape(unsigned rows, unsigned cols) {
        
        elementCount = rows * cols;
        shape = Shape2d(rows, cols);
        
        data = std::shared_ptr<T>(
            new T[elementCount], [&](T* ptr){ delete[] ptr; });
    }
}