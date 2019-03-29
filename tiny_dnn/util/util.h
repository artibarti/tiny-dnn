/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef CNN_NO_SERIALIZATION
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#endif

#include "tiny_dnn/util/config/config.h"
#include "tiny_dnn/util/error/nn_error.h"
#include "tiny_dnn/util/types/types.h"
#include "tiny_dnn/util/functions/functions.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image/image.h"
#endif

