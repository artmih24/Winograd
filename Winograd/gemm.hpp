#pragma once

#include <cstdint>
#include <immintrin.h>

#include "half.hpp"

using namespace half_float::literal;

int gemm(half_float::half* in, half_float::half* wts, half_float::half* out, int Q, int L, int F);

int optimized_gemm(half_float::half* in, half_float::half* wts, half_float::half* out, int Q, int L, int F);

int winograd_gemm(half_float::half* in, half_float::half* wts, half_float::half* out, int Q, int L, int F);