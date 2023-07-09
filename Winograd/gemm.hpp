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
template <typename T1, typename T2, typename T3>
int gemm(T1* in, T2* wts, T3* out, int Q, int L, int F);

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
template <typename T1, typename T2, typename T3>
int optimized_gemm(T1* in, T2* wts, T3* out, int Q, int L, int F);

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
template <typename T1, typename T2, typename T3>
int winograd_gemm(T1* in, T2* wts, T3* out, int Q, int L, int F);

#include "gemm.ipp"