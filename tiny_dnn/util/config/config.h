/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once


// gemmlowp doesn't support MSVC/mingw
#ifdef USE_GEMMLOWP
  #if !defined(_MSC_VER) && !defined(_WIN32) && !defined(WIN32)
    #define CNN_USE_GEMMLOWP
  #endif
#endif

// number of task in batch-gradient-descent
#ifdef CNN_USE_OMP
  #define CNN_TASK_SIZE 100
#else
  #define CNN_TASK_SIZE 8
#endif

#define CNN_UNREFERENCED_PARAMETER(x) (void)(x)

#if defined _WIN32 && !defined(__MINGW32__)
  #define CNN_WINDOWS
#endif

#if defined(_MSC_VER)
  #define CNN_MUST_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__) || defined(__ICC)
  #define CNN_MUST_INLINE __attribute__((always_inline)) inline
#else
  #define CNN_MUST_INLINE inline
#endif
