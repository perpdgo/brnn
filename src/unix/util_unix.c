/*  brnn/src/unix/util_unix.c by Paulino Perez Rodriguez
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
 
 *  Last modified: East Lansing, Michigan, Oct. 2019
 */
 


#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "util.h"
#include "Lanczos.h"

#ifdef _OPENMP
  #include <omp.h>
  //#define CSTACK_DEFNS 7
  //#include "Rinterface.h"
#endif

SEXP predictions_nn(SEXP X, SEXP n, SEXP p, SEXP theta, SEXP neurons,SEXP yhat, SEXP reqCores)
{
   int i,j,k;
   double sum,z;
   int rows, columns, nneurons;
   int useCores, haveCores;
   double *pX, *ptheta, *pyhat;
   
   SEXP list;

   rows=INTEGER_VALUE(n);
   columns=INTEGER_VALUE(p);
   nneurons=INTEGER_VALUE(neurons);

   PROTECT(X=AS_NUMERIC(X));
   pX=NUMERIC_POINTER(X);

   PROTECT(theta=AS_NUMERIC(theta));
   ptheta=NUMERIC_POINTER(theta);


   PROTECT(yhat=AS_NUMERIC(yhat));
   pyhat=NUMERIC_POINTER(yhat);

   /*
   Set the number of threads
   */
   
   #ifdef _OPENMP     
     //R_CStackLimit=(uintptr_t)-1;
     useCores=INTEGER_VALUE(reqCores);
     haveCores=omp_get_num_procs();
     if(useCores<=0 || useCores>haveCores) useCores=haveCores;
     omp_set_num_threads(useCores);   
   #endif

   #pragma omp parallel private(j,k,z,sum) 
   {
        #pragma omp for schedule(static)
   	for(i=0;i<rows;i++)
   	{
      		sum=0;
      		for(k=0;k<nneurons;k++)
      		{
	 		z=0;
	 		for(j=0;j<columns;j++)
	 		{
	    			z+=pX[i+(j*rows)]*ptheta[(columns+2)*k+j+2];
	 		}
	 		z+=ptheta[(columns+2)*k+1];      
	 		sum+=ptheta[(columns+2)*k]*tansig(z);
      		}
      		pyhat[i]=sum;
   	}
   }

   PROTECT(list=allocVector(VECSXP,1));
   SET_VECTOR_ELT(list,0,yhat);

   UNPROTECT(4);

   return(list);
}

//This function will calculate the Jocobian for the errors
SEXP jacobian_(SEXP X, SEXP n, SEXP p, SEXP theta, SEXP neurons,SEXP J, SEXP reqCores)
{
   int i,j,k;
   double z,dtansig;
   int useCores, haveCores;
   double *pX;
   double *ptheta;
   double *pJ;
   int rows, columns, nneurons;

   SEXP list;

   rows=INTEGER_VALUE(n);
   columns=INTEGER_VALUE(p);
   nneurons=INTEGER_VALUE(neurons);
  
   PROTECT(X=AS_NUMERIC(X));
   pX=NUMERIC_POINTER(X);
   
   PROTECT(theta=AS_NUMERIC(theta));
   ptheta=NUMERIC_POINTER(theta);
   
   PROTECT(J=AS_NUMERIC(J));
   pJ=NUMERIC_POINTER(J);
   
   /*
   Set the number of threads
   */

   #ifdef _OPENMP
     //R_CStackLimit=(uintptr_t)-1;
     useCores=INTEGER_VALUE(reqCores);
     haveCores=omp_get_num_procs();
     if(useCores<=0 || useCores>haveCores) useCores=haveCores;
     omp_set_num_threads(useCores); 
   #endif

   #pragma omp parallel private(j,k,z,dtansig) 
   {
        #pragma omp for schedule(static)
   	for(i=0; i<rows; i++)
   	{
                //Rprintf("i=%d\n",i);
     		for(k=0; k<nneurons; k++)
     		{
	  		z=0;
	  		for(j=0;j<columns;j++)
	  		{
	      			z+=pX[i+(j*rows)]*ptheta[(columns+2)*k+j+2]; 
	  		}
	  		z+=ptheta[(columns+2)*k+1];
	  		dtansig=pow(sech(z),2.0);
	  
	  		/*
	  		 Derivative with respect to the weight
	  		*/
	  		pJ[i+(((columns+2)*k)*rows)]=-tansig(z);
	 
	  		/*
	  		Derivative with respect to the bias
	 		*/
	 
	 		pJ[i+(((columns+2)*k+1)*rows)]=-ptheta[(columns+2)*k]*dtansig;

	 		/*
	  		 Derivate with respect to the betas
	  		*/
	 		for(j=0; j<columns;j++)
	 		{
	     			pJ[i+(((columns+2)*k+j+2)*rows)]=-ptheta[(columns+2)*k]*dtansig*pX[i+(j*rows)];
	 		}
     		}
   	}
   }
  
   PROTECT(list=allocVector(VECSXP,1));
   SET_VECTOR_ELT(list,0,J);
   
   UNPROTECT(4);
   
   return(list);
}

SEXP estimate_trace(SEXP A, SEXP n, SEXP lambdamin, SEXP lambdamax, SEXP tol, SEXP samples, SEXP reqCores, SEXP rz, SEXP ans)
{ 
   int useCores, haveCores;
   int i;
   int nsamples;
   double lmin, lmax;
   double max_error;
   int rows;
   double *pA;
   double *pans;
   double *prz;
   double sum=0;
   
   
   SEXP list;
   
   nsamples=INTEGER_VALUE(samples);
   lmin=NUMERIC_VALUE(lambdamin);
   lmax=NUMERIC_VALUE(lambdamax);
   rows=INTEGER_VALUE(n);
   max_error=NUMERIC_VALUE(tol);
   useCores=INTEGER_VALUE(reqCores);
   
   PROTECT(A=AS_NUMERIC(A));
   pA=NUMERIC_POINTER(A);
   
   PROTECT(ans=AS_NUMERIC(ans));
   pans=NUMERIC_POINTER(ans);
   
   PROTECT(rz=AS_NUMERIC(rz));
   prz=NUMERIC_POINTER(rz);
   
   /*
   Set the number of threads
   */
   
   #ifdef _OPENMP
     //R_CStackLimit=(uintptr_t)-1;
     haveCores=omp_get_num_procs();
     if(useCores<=0 || useCores>haveCores) useCores=haveCores;
     omp_set_num_threads(useCores);
   #endif 
   
   #pragma omp parallel
   {
      /*Starts the work sharing construct*/
      #pragma omp for reduction(+:sum) schedule(static)
      for(i=0; i<nsamples; i++)
      {
	sum+=Bai(pA,&rows,&lmin, &lmax, &max_error,prz,&i);
      }
   }
   
   *pans=(sum/(nsamples));
   
   PROTECT(list=allocVector(VECSXP,1));
   SET_VECTOR_ELT(list,0,ans);
   
   UNPROTECT(4);
   
   return(list);
}

/*
 Computes the inverse of a triangular matrix obtained by using Cholesky
 decomposition. 
 
 Let: 

 AA'= B
 
 We want to obtain the trace(B^{-1})

 Algorithm:

 a) Compute A
 b) Compute A^{-1}, this is because B^{-1}={AA'}^{-1}={A'}^{-1} A^{-1}
 c) Compute the trace obtaining only the elements of the diagonal of B^{-1} and then take the sum. This saves
    some computations since we do not need to obtain all elements.
 */

SEXP La_dtrtri_(SEXP A, SEXP size)
{
    int sz = asInteger(size);
    if (sz == NA_INTEGER || sz < 1) {
	error("size argument must be a positive integer");
	return R_NilValue; /* -Wall */
    } else {
	SEXP ans, Amat = A; /* -Wall: we initialize here as for the 1x1 case */
	int m = 1, n = 1, nprot = 0;

	if (sz == 1 && !isMatrix(A) && isReal(A)) {
	    /* nothing to do; m = n = 1; ... */
	} else if (isMatrix(A)) {
	    SEXP adims = getAttrib(A, R_DimSymbol);
	    if (TYPEOF(adims) != INTSXP) error("non-integer dims");
	    Amat = PROTECT(coerceVector(A, REALSXP)); nprot++;
	    m = INTEGER(adims)[0]; n = INTEGER(adims)[1];
	} else error("a must be a numeric matrix");

	if (sz > n) { UNPROTECT(nprot); error("size cannot exceed ncol(x) = %d", n); }
	if (sz > m) { UNPROTECT(nprot); error("size cannot exceed nrow(x) = %d", m); }
	ans = PROTECT(allocMatrix(REALSXP, sz, sz)); nprot++;
	size_t M = m, SZ = sz;
	for (int j = 0; j < sz; j++) {
	    for (int i = 0; i <= j; i++)
		REAL(ans)[i + j * SZ] = REAL(Amat)[i + j * M];
	}
	
	int info;
	
	F77_CALL(dtrtri)("Upper", "Non-unit", &sz, REAL(ans), &sz, &info);
	
	
	if (info != 0) {
	    UNPROTECT(nprot);
	    if (info > 0)
		error("element (%d, %d) is zero, so the inverse cannot be computed",
		      info, info);
	    error("argument %d of Lapack routine %s had invalid value",
		  -info, "dtrtri");
	}
	
	
	/*
	The elements in lower triangular are set to 0
	*/
	/*
	for (int j = 0; j < sz; j++)
	    for (int i = j+1; i < sz; i++)
		REAL(ans)[i + j * SZ] = 0.0;
	
	*/
	
	double sum=0;
	
	SEXP trace=PROTECT(allocVector(REALSXP,1));
	
	for (int j = 0; j < sz; j++) 
	{
	    for (int i = 0; i <= j; i++)
	    {
	    	sum=sum+REAL(ans)[i + j * SZ]*REAL(ans)[i + j * SZ];
	    }
	}
	
	REAL(trace)[0]=sum;
	
	/*
	Rprintf("sum=%f\n",sum);
	*/
	
	/*UNPROTECT(nprot);*/
	
	UNPROTECT(3);
	
	
	/*return ans;*/
	
	return(trace);
	
    }
}
