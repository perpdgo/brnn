/* 
  file brnn/src/nominal_core.c

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 or 3 of the License
  (at your option).

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  A copy of the GNU General Public License is available at
  http://www.r-project.org/Licenses/

  A package for Bayesian Regularized Neural Networks
  Author: Elmer Nava Sanchez

  Texcoco, Mexico, Sep. 2026

  WARNING: This is an experimental version
*/

#include <R.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/RS.h>
#include <math.h>
#include <string.h>

/*
 * Forward propagation for a neural network for norminal response with only one hidden layer
 *
 * Matrices are stored in R by column:
 *
 *   X       : n x p
 *   W_out   : s x K
 *   Lambda  : p x s
 *
 * Codes for activation functions:
 *
 *   0 = tanh
 *   1 = sigmoid
 *   2 = relu
 */
SEXP forward_pass_nominal(SEXP X,
                          SEXP n_,
                          SEXP p_,
                          SEXP W_out,
                          SEXP b_hidden,
                          SEXP Lambda,
                          SEXP b_out,
                          SEXP neurons_,
                          SEXP K_,
                          SEXP activation_,
                          SEXP U,
                          SEXP H,
                          SEXP O,
                          SEXP P)
{
    int i, j, k, l;

    int n = asInteger(n_);
    int p = asInteger(p_);
    int s = asInteger(neurons_);
    int K = asInteger(K_);
    int activation = asInteger(activation_);

    double value;
    double max_logit;
    double row_sum;

    if (n < 1 || p < 1 || s < 1 || K < 2) {
        error("Invalid dimmensions in routine forward_pass_nominal.");
    }

    if (activation < 0 || activation > 2) {
        error("Invalid code for selecting activation function.");
    }

    PROTECT(X = coerceVector(X, REALSXP));
    PROTECT(W_out = coerceVector(W_out, REALSXP));
    PROTECT(b_hidden = coerceVector(b_hidden, REALSXP));
    PROTECT(Lambda = coerceVector(Lambda, REALSXP));
    PROTECT(b_out = coerceVector(b_out, REALSXP));

    PROTECT(U = coerceVector(U, REALSXP));
    PROTECT(H = coerceVector(H, REALSXP));
    PROTECT(O = coerceVector(O, REALSXP));
    PROTECT(P = coerceVector(P, REALSXP));

    double *pX = REAL(X);
    double *pW_out = REAL(W_out);
    double *pb_hidden = REAL(b_hidden);
    double *pLambda = REAL(Lambda);
    double *pb_out = REAL(b_out);

    double *pU = REAL(U);
    double *pH = REAL(H);
    double *pO = REAL(O);
    double *pP = REAL(P);

    /*
     * Inputs in the hidden layer:
     *
     * U = X %*% Lambda + b_hidden
     *
     * Activation function:
     *
     * H = g(U)
     */
    for (k = 0; k < s; k++) {

        for (i = 0; i < n; i++) {

            value = pb_hidden[k];

            for (j = 0; j < p; j++) {
                value +=
                    pX[i + j * n] *
                    pLambda[j + k * p];
            }

            pU[i + k * n] = value;

            if (activation == 0) {

                pH[i + k * n] = tanh(value);

            } else if (activation == 1) {

                pH[i + k * n] =
                    1.0 / (1.0 + exp(-value));

            } else {

                pH[i + k * n] =
                    (value > 0.0) ? value : 0.0;
            }
        }
    }

    /*
     * Logits:
     *
     * O = H %*% W_out + b_out
     */
    for (l = 0; l < K; l++) {

        for (i = 0; i < n; i++) {

            value = pb_out[l];

            for (k = 0; k < s; k++) {
                value +=
                    pH[i + k * n] *
                    pW_out[k + l * s];
            }

            pO[i + l * n] = value;
        }
    }

    /*
     * Numerically stable softmax function, computed row by row.
     */
    for (i = 0; i < n; i++) {

        max_logit = pO[i];

        for (l = 1; l < K; l++) {

            if (pO[i + l * n] > max_logit) {
                max_logit = pO[i + l * n];
            }
        }

        row_sum = 0.0;

        for (l = 0; l < K; l++) {

            value = exp(
                pO[i + l * n] - max_logit
            );

            pP[i + l * n] = value;
            row_sum += value;
        }

        for (l = 0; l < K; l++) {
            pP[i + l * n] /= row_sum;
        }
    }

    SEXP result;
    SEXP names;

    PROTECT(result = allocVector(VECSXP, 4));
    PROTECT(names = allocVector(STRSXP, 4));

    SET_VECTOR_ELT(result, 0, U);
    SET_VECTOR_ELT(result, 1, H);
    SET_VECTOR_ELT(result, 2, O);
    SET_VECTOR_ELT(result, 3, P);

    SET_STRING_ELT(
        names,
        0,
        mkChar("U")
    );

    SET_STRING_ELT(
        names,
        1,
        mkChar("H")
    );

    SET_STRING_ELT(
        names,
        2,
        mkChar("O")
    );

    SET_STRING_ELT(
        names,
        3,
        mkChar("P")
    );

    setAttrib(
        result,
        R_NamesSymbol,
        names
    );

    UNPROTECT(11);

    return result;
}


/*
 * Fisher/Gauss-Newton approximation:
 *
 * H_data = suma_i J_i' S_i J_i
 *
 * where:
 *
 * S_i = diag(p_i) - p_i p_i'
 *
 * The order of the parameters coincides with that generated with the function flatten_theta() in R:
 *
 *   1. W_out, by columns
 *   2. b_hidden
 *   3. Lambda, by colums
 *   4. b_out
 *
 * matrix products are performed using BLAS.
 */
SEXP hessian_approx_nominal(SEXP X,
                            SEXP P,
                            SEXP U,
                            SEXP H_hidden,
                            SEXP W_out,
                            SEXP Lambda,
                            SEXP n_,
                            SEXP p_,
                            SEXP neurons_,
                            SEXP K_,
                            SEXP activation_,
                            SEXP H_data)
{
    int i, j, k, l, m;
    int idx;

    int n = asInteger(n_);
    int p = asInteger(p_);
    int s = asInteger(neurons_);
    int K = asInteger(K_);
    int activation = asInteger(activation_);

    int q = s * K + s + p * s + K;

    double one = 1.0;
    double zero = 0.0;

    if (n < 1 ||
        p < 1 ||
        s < 1 ||
        K < 2 ||
        q < 1) {

        error(
            "Invalid dimmensions in routine "
            "hessian_approx_nominal."
        );
    }

    if (activation < 0 || activation > 2) {
        error("Invalid code for selecting activation function.");
    }

    PROTECT(X = coerceVector(X, REALSXP));
    PROTECT(P = coerceVector(P, REALSXP));
    PROTECT(U = coerceVector(U, REALSXP));
    PROTECT(H_hidden = coerceVector(H_hidden, REALSXP));
    PROTECT(W_out = coerceVector(W_out, REALSXP));
    PROTECT(Lambda = coerceVector(Lambda, REALSXP));
    PROTECT(H_data = coerceVector(H_data, REALSXP));

    /*
     * Checking dimmensions
     */
    if (XLENGTH(X) != (R_xlen_t) n * p ||
        XLENGTH(P) != (R_xlen_t) n * K ||
        XLENGTH(U) != (R_xlen_t) n * s ||
        XLENGTH(H_hidden) != (R_xlen_t) n * s ||
        XLENGTH(W_out) != (R_xlen_t) s * K ||
        XLENGTH(Lambda) != (R_xlen_t) p * s ||
        XLENGTH(H_data) != (R_xlen_t) q * q) {

        UNPROTECT(7);

        error(
            "Incorrect dimmensiones in routine "
            "hessian_approx_nominal."
        );
    }

    double *pX = REAL(X);
    double *pP = REAL(P);
    double *pU = REAL(U);
    double *pH_hidden = REAL(H_hidden);
    double *pW_out = REAL(W_out);
    double *pH_data = REAL(H_data);

    /*
     * Temporary memory controlled by R
     *
     * gp : derivative of the activation function, length s
     * J  : Jacobian K x q
     * S  : curvature matrix K x K
     * SJ : product S x J, dimmensions K x q
     */
    double *gp =
        (double *) R_alloc(
            (size_t) s,
            sizeof(double)
        );

    double *J =
        (double *) R_alloc(
            (size_t) K * q,
            sizeof(double)
        );

    double *S =
        (double *) R_alloc(
            (size_t) K * K,
            sizeof(double)
        );

    double *SJ =
        (double *) R_alloc(
            (size_t) K * q,
            sizeof(double)
        );

    /*
     * Lambda is taken as argument to keep the same conceptual interface
     * of a previous implementation in C++. The values are not used directly 
     * here because U was already computed during the propagation 
    */

    (void) REAL(Lambda);

    /*
     * Inicializar the Hessian matrix
     */
    memset(
        pH_data,
        0,
        (size_t) q * q * sizeof(double)
    );

    /*
     * Loop over observations
     */
    for (i = 0; i < n; i++) {

        /*
         * Derivative of the activation function evaluated in U[i,].
         */
        for (k = 0; k < s; k++) {

            double u =
                pU[i + k * n];

            if (activation == 0) {

                double t = tanh(u);
                gp[k] = 1.0 - t * t;

            } else if (activation == 1) {

                double sig =
                    1.0 / (1.0 + exp(-u));

                gp[k] =
                    sig * (1.0 - sig);

            } else {

                gp[k] =
                    (u > 0.0) ? 1.0 : 0.0;
            }
        }

        /*
         * Initializing Jacobian J_i in zero
         *
         * J_i has dimmensions K x q and is stored
         * by columns, similar to R matrices
         */
        memset(
            J,
            0,
            (size_t) K * q * sizeof(double)
        );

        idx = 0;

        /*
         * 1. Derivatives with respect to W_out
         *
         * W_out is vectorized by columns
         */
        for (l = 0; l < K; l++) {

            for (k = 0; k < s; k++) {

                J[l + idx * K] =
                    pH_hidden[i + k * n];

                idx++;
            }
        }

        /*
         * 2. Derivatives with respect to b_hidden.
         */
        for (k = 0; k < s; k++) {

            for (l = 0; l < K; l++) {

                J[l + idx * K] =
                    pW_out[k + l * s] *
                    gp[k];
            }

            idx++;
        }

        /*
         * 3. Derivatives with respect to Lambda
         *
         * Lambda is vectorized by columns
         */
        for (k = 0; k < s; k++) {

            for (j = 0; j < p; j++) {

                for (l = 0; l < K; l++) {

                    J[l + idx * K] =
                        pW_out[k + l * s] *
                        gp[k] *
                        pX[i + j * n];
                }

                idx++;
            }
        }

        /*
         * 4. Derivatives respect to b_out
         */
        for (l = 0; l < K; l++) {

            J[l + idx * K] = 1.0;
            idx++;
        }

        /*
         * Check that the q parameters were visited correctly
         */
        if (idx != q) {

            UNPROTECT(7);

            error(
                "Inconsistent parameters "
                "internal order."
            );
        }

        /*
         * Soft max curvature matrix:
         *
         * S_i = diag(p_i) - p_i p_i'
         */
        for (m = 0; m < K; m++) {

            for (l = 0; l < K; l++) {

                double p_l =
                    pP[i + l * n];

                double p_m =
                    pP[i + m * n];

                S[l + m * K] =
                    ((l == m) ? p_l : 0.0) -
                    p_l * p_m;
            }
        }

        /*
         * SJ = S_i %*% J_i
         *
         * S  : K x K
         * J  : K x q
         * SJ : K x q
         */
        F77_CALL(dgemm)(
            "N",
            "N",
            &K,
            &q,
            &K,
            &one,
            S,
            &K,
            J,
            &K,
            &zero,
            SJ,
            &K
            FCONE FCONE
        );

        /*
         * H_data += t(J_i) %*% SJ
         *
         * t(J_i) : q x K
         * SJ     : K x q
         * H_data : q x q
         */
        F77_CALL(dgemm)(
            "T",
            "N",
            &q,
            &q,
            &K,
            &one,
            J,
            &K,
            SJ,
            &K,
            &one,
            pH_data,
            &q
            FCONE FCONE
        );
    }

    UNPROTECT(7);

    return H_data;
}
