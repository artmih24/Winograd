#include "gemm.hpp"

int gemm(half_float::half* in, half_float::half* wts, half_float::half* out, int Q, int L, int F) {
    for (int q = 0; q < Q; q++)
        for (int l = 0; l < L; l++)
            for (int f = 0; f < F; f++)
                out[q * F + f] += in[q * L + l] * wts[l * F + f];
	return 0;
}

int winograd_gemm(half_float::half* in, half_float::half* wts, half_float::half* out, int Q, int L, int F) {
    int hL = L / 2;
    half_float::half *rowFactors = new half_float::half[Q],
                     *colFactors = new half_float::half[F];

    // compute row factors
    for (int q = 0; q < Q; q++) {
        rowFactors[q] = in[q * L] * in[q * L + 1];
        for (int hl = 1; hl < hL; hl++)
            rowFactors[q] += in[q * L + 2 * hl] * in[q * L + 2 * hl + 1];
    }

    // compute column factors
    for (int f = 0; f < F; f++) {
        colFactors[f] = wts[f] * wts[F + f];
        for (int hl = 1; hl < hL; hl++)
            colFactors[f] += wts[(2 * hl) * F + f] * wts[(2 * hl + 1) * F + f];
    }

    // compute output matrix
    for (int q = 0; q < Q; q++)
        for (int f = 0; f < F; f++) {
            out[q * F + f] = -rowFactors[q] - colFactors[f];
            for (int hl = 0; hl < hL; hl++)
                out[q * F + f] += (in[q * L + 2 * hl] + wts[(2 * hl + 1) * F + f]) * (in[q * L + 2 * hl + 1] + wts[(2 * hl) * F + f]);
        }

    // add last items if L is odd
    if (L % 2 == 1)
        for (int q = 0; q < Q; q++)
            for (int f = 0; f < F; f++)
                out[q * F + f] += in[q * L + (L - 1)] * wts[(L - 1) * F + f];

    delete[] rowFactors;
    delete[] colFactors;
    return 0;
}
