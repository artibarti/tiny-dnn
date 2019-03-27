/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cstddef>
#include <cstdint>

#define CNN_USE_STDOUT

#ifdef USE_GEMMLOWP
  #if !defined(_MSC_VER) && !defined(_WIN32) && !defined(WIN32)
    #define CNN_USE_GEMMLOWP  // gemmlowp doesn't support MSVC/mingw
  #endif
#endif

/**
 * number of task in batch-gradient-descent.
 * @todo automatic optimization
 */
#ifdef CNN_USE_OMP
  #define CNN_TASK_SIZE 100
#else
  #define CNN_TASK_SIZE 8
#endif