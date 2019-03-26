#pragma once

#include <iostream>

namespace tiny_dnn
{
    struct Shape2d
    {
        int x, y;
        
        Shape2d() : x(0), y(0) {}
        Shape2d(int _x, int _y) : x(_x), y(_y) {}
                
        bool operator==(Shape2d _other) {
            return (x == _other.x && y == _other.y);
        }
        
        bool operator!=(Shape2d _other) {
            return !(x == _other.x && y == _other.y);
        }

        int operator[] (unsigned index) {
            if (index == 0) {
                return x;
            }
            else if (index == 1) {
                return y;
            }
            else {
                throw std::out_of_range("Index out of bounds");
            }
        }
    };
}