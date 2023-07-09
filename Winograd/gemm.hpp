#pragma once

#include "half.hpp"

using namespace half_float::literal;

/// <summary>
/// matrix multiplication
/// </summary>
/// <param name="in">- input matrix</param>
/// <param name="wts">- filters matrix</param>
/// <param name="out">- output matrix</param>
/// <param name="Q">- lines of input and output matrices</param>
/// <param name="L">- columns of input matrix and lines of filters matrix</param>
/// <param name="F">- columns of filters and output matrices</param>
/// <returns>0</returns>
int gemm(half_float::half* in, half_float::half* wts, half_float::half* out, int Q, int L, int F);

/// <summary>
/// oprimized matrix multiplication
/// </summary>
/// <param name="in">- input matrix</param>
/// <param name="wts">- filters matrix</param>
/// <param name="out">- output matrix</param>
/// <param name="Q">- lines of input and output matrices</param>
/// <param name="L">- columns of input matrix and lines of filters matrix</param>
/// <param name="F">- columns of filters and output matrices</param>
/// <returns>0</returns>
int optimized_gemm(half_float::half* in, half_float::half* wts, half_float::half* out, int Q, int L, int F);

/// <summary>
/// matrix multiplication with using Winograd algorithm
/// </summary>
/// <param name="in">- input matrix</param>
/// <param name="wts">- filters matrix</param>
/// <param name="out">- output matrix</param>
/// <param name="Q">- lines of input and output matrices</param>
/// <param name="L">- columns of input matrix and lines of filters matrix</param>
/// <param name="F">- columns of filters and output matrices</param>
/// <returns>0</returns>
int winograd_gemm(half_float::half* in, half_float::half* wts, half_float::half* out, int Q, int L, int F);