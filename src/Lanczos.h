/*  brnn/src/Lanczos.h
 *
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 or 3 of the License
 *  (at your option).
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  A copy of the GNU General Public License is available at
 *  http://www.r-project.org/Licenses/
 */


#include <stdlib.h>
#include <math.h>
#include <R.h>
#include <R_ext/Lapack.h>

void mgcv_trisymeig(double *d,double *g,double *v,int *n,int getvec,int descending);
void extreme_eigenvalues(double *A,double *U,double *D,int *n, int *m, int *lm,double *tol);
double Bai(double *A,int *n,double *lambdamin, double *lambdamax, double *tol, double *rz, int *col); 
