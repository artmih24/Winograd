#include <iostream>
#include <cstdint>
#include <chrono>

#include "half.hpp"
#include "gemm.hpp"
#include "jslike.h"

using namespace half_float::literal;
using namespace jslike;

#define MATRIX_PRINT

int main() {
    #ifdef _MSC_VER
        system("chcp 1253 > nul");
    #endif
    //half_float::half a = 0.123_h;
    //std::cout << "a = " << a << " = 0x" << std::hex << *(uint16_t*)(&a) << std::dec << std::endl;
    int Q = 16,
        L = 16,
        F = 16;
    size_t QL = static_cast<size_t>(Q) * L,
           LF = static_cast<size_t>(L) * F,
           QF = static_cast<size_t>(Q) * F;
    half_float::half *in      = new half_float::half[QL],
                     *wts     = new half_float::half[LF],
                     *out     = new half_float::half[QF],
                     *opt_out = new half_float::half[QF],
                     *win_out = new half_float::half[QF];
    for (int q = 0; q < Q; q++)
        for (int l = 0; l < L; l++)
            in[q * L + l] = static_cast<half_float::half>((1 + 2 * q + l) % 10);
    for (int l = 0; l < L; l++)
        for (int f = 0; f < F; f++)
            wts[l * F + f] = static_cast<half_float::half>((5 + 3 * l + f) % 10);
    for (int q = 0; q < Q; q++)
        for (int f = 0; f < F; f++) {
            out[q * F + f]     = 0.0_h;
            win_out[q * F + f] = 0.0_h;
        }
    #ifdef MATRIX_PRINT
        //std::cout << std::endl;
        for (int q = 0; q < Q; q++) {
            for (int l = 0; l < L; l++)
                std::cout << in[q * L + l] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for (int l = 0; l < L; l++) {
            for (int f = 0; f < F; f++)
                std::cout << wts[l * F + f] << " ";
            std::cout << std::endl;
        }
    #endif // MATRIX_PRINT

    auto start = std::chrono::steady_clock::now();
    gemm(in, wts, out, Q, L, F);
    auto end = std::chrono::steady_clock::now();

    #ifdef MATRIX_PRINT
        std::cout << std::endl;
        for (int q = 0; q < Q; q++) {
            for (int f = 0; f < F; f++)
                std::cout << out[q * F + f] << " ";
            std::cout << std::endl;
        }
    #endif

    auto optimized_start = std::chrono::steady_clock::now();
    optimized_gemm(in, wts, opt_out, Q, L, F);
    auto optimized_end = std::chrono::steady_clock::now();

    #ifdef MATRIX_PRINT
        std::cout << std::endl;
        for (int q = 0; q < Q; q++) {
            for (int f = 0; f < F; f++)
                std::cout << opt_out[q * F + f] << " ";
            std::cout << std::endl;
        }
    #endif

    auto win_start = std::chrono::steady_clock::now();
    winograd_gemm(in, wts, win_out, Q, L, F);
    auto win_end = std::chrono::steady_clock::now();

    #ifdef MATRIX_PRINT
        std::cout << std::endl;
        for (int q = 0; q < Q; q++) {
            for (int f = 0; f < F; f++)
                std::cout << win_out[q * F + f] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    #endif

    std::cout << "gemm:" << std::endl;
    std::cout << "Elapsed time in nanoseconds:  " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << " ns" << std::endl;
    std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " µs" << std::endl;
    std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "Elapsed time in seconds:      " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " sec" << std::endl;

    std::cout << std::endl;

    std::cout << "gemm:" << std::endl;
    std::cout << "Elapsed time in nanoseconds:  " << std::chrono::duration_cast<std::chrono::nanoseconds>(optimized_end - optimized_start).count() << " ns" << std::endl;
    std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds>(optimized_end - optimized_start).count() << " µs" << std::endl;
    std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(optimized_end - optimized_start).count() << " ms" << std::endl;
    std::cout << "Elapsed time in seconds:      " << std::chrono::duration_cast<std::chrono::seconds>(optimized_end - optimized_start).count() << " sec" << std::endl;

    std::cout << std::endl;

    std::cout << "winograd gemm:" << std::endl;
    std::cout << "Elapsed time in nanoseconds:  " << std::chrono::duration_cast<std::chrono::nanoseconds>(win_end - win_start).count() << " ns" << std::endl;
    std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds>(win_end - win_start).count() << " µs" << std::endl;
    std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(win_end - win_start).count() << " ms" << std::endl;
    std::cout << "Elapsed time in seconds:      " << std::chrono::duration_cast<std::chrono::seconds>(win_end - win_start).count() << " sec" << std::endl;

    delete[] in;
    delete[] wts;
    delete[] out;
    delete[] opt_out;
    delete[] win_out;
    #ifdef _MSC_VER
    system("chcp 1251 > nul");
    #endif
    return 0;
}
