template<typename T1, typename T2, typename T3>
int gemm(T1* in, T2* wts, T3* out, int Q, int L, int F) {
    for (int q = 0; q < Q; q++)
        for (int l = 0; l < L; l++)
            for (int f = 0; f < F; f++)
                out[q * F + f] += in[q * L + l] * wts[l * F + f];
    return 0;
}

template<typename T1, typename T2, typename T3>
int optimized_gemm(T1* in, T2* wts, T3* out, int Q, int L, int F) {
    int q,                      // iteration variable for Q loop
        l,                      // iteration variable for L loop
        f,                      // iteration variable for F loop
        qL,                     // temporary variable for q * L value
        lF,                     // temporary variable for l * F value
        qF;                     // temporary variable for q * F value
    T1 in_qLl;                  // temporary variable for in[q * L + l] value
    for (q = 0; q < Q; q++) {
        qL = q * L;
        qF = q * F;
        for (l = 0; l < L; l++) {
            lF = l * F;
            in_qLl = in[qL + l];
            for (f = 0; f < F; f += 4) {
                out[qF + f] += in_qLl * wts[lF + f];
                out[qF + f + 1] += in_qLl * wts[lF + f + 1];
                out[qF + f + 2] += in_qLl * wts[lF + f + 2];
                out[qF + f + 3] += in_qLl * wts[lF + f + 3];
            }
        }
    }
    return 0;
}

template<typename T1, typename T2, typename T3>
int winograd_gemm(T1* in, T2* wts, T3* out, int Q, int L, int F) {
    int hL = L / 2;                                             // L / 2 value
    T3 *rowFactors = new T3[Q],     // row factors array
       *colFactors = new T3[F];     // column factors array

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
