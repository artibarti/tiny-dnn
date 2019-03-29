#pragma once

#include <iostream>

namespace tiny_dnn {

struct index2d
{
    unsigned x, y;

    index2d() : x(0), y(0) {}
    index2d(unsigned _x, unsigned _y) : x(_x), y(_y) {}
            
    bool operator==(const index2d& other) {
        return (x == other.x && y == other.y);
    }    
    bool operator!=(const index2d& other) {
        return !(x == other.x && y == other.y);
    }
    bool operator==(unsigned size) {// 
        return (x == size && y == size);
    }    

    unsigned operator[] (unsigned index) {
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else {
            throw std::out_of_range("Index out of bounds");
        }
    }
};

using shape2d = index2d;

}