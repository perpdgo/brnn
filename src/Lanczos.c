/*  brnn/src/Lanczos.c
 *
 *  mgcv_trisymeig and extreme_eigenvalues adapted from the routines
 *  ggcv_trisymeig and Rlanczos routines in the mgcv R package, Version 1.7-22, http://cran.r-project.org/web/packages/mgcv/index.html
 *  
 *  routine Bai implements the Bai's algorithm, 
 *
 *  Bai, Z. J., M. Fahey and G. Golub (1996). Some large-scale matrix 
 *  computation problems. Journal of Computational and Applied Mathematics 74(1-2): 71-89. 
 *
 *  Bai's algorithm implemented by Paulino Perez Rodriguez  
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


#include "Lanczos.h"

void mgcv_trisymeig(double *d,double *g,double *v,int *n,int getvec,int descending) 
/* Find eigen-values and vectors of n by n symmetric tridiagonal matrix 
   with leading diagonal d and sub/super diagonals g. 
   eigenvalues returned in d, and eigenvectors in columns of v, if
   getvec!=0. If *descending!=0 then eigenvalues returned in descending order,
   otherwise ascending. eigen-vector order corresponds.  

   Routine is divide and conquer followed by inverse iteration. 

   dstevd could be used instead, with just a name change.
   dstevx may be faster, but needs argument changes.
*/ 
		    
{ char compz;
  double *work,work1,x,*dum1,*dum2;
  int ldz=0,info,lwork=-1,liwork=-1,*iwork,iwork1,i,j;

  if (getvec) { compz='I';ldz = *n;} else { compz='N';ldz=0;}

  /* workspace query first .... */
  F77_NAME(dstedc)(&compz,n,
		   d, g, /* lead and su-diag */
		   v, /* eigenvectors on exit */  
                   &ldz, /* dimension of v */
		   &work1, &lwork,
		   &iwork1, &liwork, &info);

   lwork=(int)floor(work1);if (work1-lwork>0.5) lwork++;
   work=(double *)calloc((size_t)lwork,sizeof(double));
   liwork = iwork1;
   iwork= (int *)calloc((size_t)liwork,sizeof(int));

   /* and the actual call... */
   F77_NAME(dstedc)(&compz,n,
		   d, g, /* lead and su-diag */
		   v, /* eigenvectors on exit */  
                   &ldz, /* dimension of v */
		   work, &lwork,
		   iwork, &liwork, &info);

   if (descending) { /* need to reverse eigenvalues/vectors */
     for (i=0;i<*n/2;i++) { /* reverse the eigenvalues */
       x = d[i]; d[i] = d[*n-i-1];d[*n-i-1] = x;
       dum1 = v + *n * i;dum2 = v + *n * (*n-i-1); /* pointers to heads of cols to exchange */
       for (j=0;j<*n;j++,dum1++,dum2++) { /* work down columns */
         x = *dum1;*dum1 = *dum2;*dum2 = x;
       }
     }
   }

   free(work);
   free(iwork);
   *n=info; /* zero is success */
}

void extreme_eigenvalues(double *A,double *U,double *D,int *n, int *m, int *lm,double *tol) {
/* Faster version of lanczos_spd for calling from R.
   A is n by n symmetric matrix. Let k = m + max(0,lm).
   U is n by k and D is a k-vector.
   m is the number of upper eigenvalues required and lm the number of lower.
   If lm<0 then the m largest magnitude eigenvalues (and their eigenvectors)
   are returned 

   Matrices are stored in R (and LAPACK) format (1 column after another).

   ISSUE: 1. Currently all eigenvectors of Tj are found, although only the next unconverged one
             is really needed. Might be better to be more selective using dstein from LAPACK. 
          2. Basing whole thing on dstevx might be faster
          3. Is random start vector really best? Actually Demmel (1997) suggests using a random vector, 
             to avoid any chance of orthogonality with an eigenvector!
          4. Could use selective orthogonalization, but cost of full orth is only 2nj, while n^2 of method is
             unavoidable, so probably not worth it.  
*/
  //Rprintf("Initializing...\n");
  int biggest=0,f_check,i,k,kk,ok,l,j,vlength=0,ni,pi,converged,incx=1;
  double **q,*v=NULL,bt,xx,yy,*a,*b,*d,*g,*z,*err,*p0,*p1,*zp,*qp,normTj,eps_stop,max_err,alpha=1.0,beta=0.0;
  unsigned long jran=1,ia=106,ic=1283,im=6075; /* simple RNG constants */
  const char uplo='U';
  eps_stop = *tol; 

  if (*lm<0) { biggest=1;*lm=0;} /* get m largest magnitude eigen-values */
  f_check = (*m + *lm)/2; /* how often to get eigen_decomp */
  if (f_check<10) f_check =10;
  kk = (int) floor(*n/10); if (kk<1) kk=1;  
  if (kk<f_check) f_check = kk;

  q=(double **)calloc((size_t)(*n+1),sizeof(double *));

  /* "randomly" initialize first q vector */
  q[0]=(double *)calloc((size_t)*n,sizeof(double));
  b=q[0];bt=0.0;
  for (i=0;i < *n;i++)   /* somewhat randomized start vector!  */
  { 
    jran=(jran*ia+ic) % im;   /* quick and dirty generator to avoid too regular a starting vector */
    xx=(double) jran / (double) im -0.5; 
    b[i]=xx;
    xx=-xx;
    bt+=b[i]*b[i];
  } 
  bt=sqrt(bt); 
  for (i=0;i < *n;i++) {
    b[i]/=bt;
    //Rprintf("b[i]=%f\n",b[i]);
  }

  /* initialise vectors a and b - a is leading diagonal of T, b is sub/super diagonal */
  a=(double *)calloc((size_t) *n,sizeof(double));
  b=(double *)calloc((size_t) *n,sizeof(double));
  g=(double *)calloc((size_t) *n,sizeof(double));
  d=(double *)calloc((size_t) *n,sizeof(double)); 
  z=(double *)calloc((size_t) *n,sizeof(double));
  err=(double *)calloc((size_t) *n,sizeof(double));
  for (i=0;i< *n;i++) err[i]=1e300;
  /* The main loop. Will break out on convergence. */
  for (j=0;j< *n;j++) 
  { /* form z=Aq[j]=A'q[j], the O(n^2) step ...  */
    /*for (Ap=A,zp=z,p0=zp+*n;zp<p0;zp++) 
      for (*zp=0.0,qp=q[j],p1=qp+*n;qp<p1;qp++,Ap++) *zp += *Ap * *qp;*/
    /*  BLAS versions y := alpha*A*x + beta*y, */
    F77_NAME(dsymv)(&uplo,n,&alpha,
		A,n,
		q[j],&incx,
		&beta,z,&incx);
    /* Now form a[j] = q[j]'z.... */
    for (xx=0.0,qp=q[j],p0=qp+*n,zp=z;qp<p0;qp++,zp++) xx += *qp * *zp;
    a[j] = xx;
 
    /* Update z..... */
    if (!j)
    { /* z <- z - a[j]*q[j] */
      for (zp=z,p0=zp+*n,qp=q[j];zp<p0;qp++,zp++) *zp -= xx * *qp;
    } else
    { /* z <- z - a[j]*q[j] - b[j-1]*q[j-1] */
      yy=b[j-1];      
      for (zp=z,p0=zp + *n,qp=q[j],p1=q[j-1];zp<p0;zp++,qp++,p1++) *zp -= xx * *qp + yy * *p1;    
  
      /* Now stabilize by full re-orthogonalization.... */
      
      for (i=0;i<=j;i++) 
      { /* form xx= z'q[i] */
        /*for (xx=0.0,qp=q[i],p0=qp + *n,zp=z;qp<p0;zp++,qp++) xx += *zp * *qp;*/
        xx = -F77_NAME(ddot)(n,z,&incx,q[i],&incx); /* BLAS version */
        /* z <- z - xx*q[i] */
        /*for (qp=q[i],zp=z;qp<p0;qp++,zp++) *zp -= xx * *qp;*/
        F77_NAME(daxpy)(n,&xx,q[i],&incx,z,&incx); /* BLAS version */
      } 
      
      /* exact repeat... */
      for (i=0;i<=j;i++) 
      { /* form xx= z'q[i] */
        /* for (xx=0.0,qp=q[i],p0=qp + *n,zp=z;qp<p0;zp++,qp++) xx += *zp * *qp; */
        xx = -F77_NAME(ddot)(n,z,&incx,q[i],&incx); /* BLAS version */
        /* z <- z - xx*q[i] */
        /* for (qp=q[i],zp=z;qp<p0;qp++,zp++) *zp -= xx * *qp; */
        F77_NAME(daxpy)(n,&xx,q[i],&incx,z,&incx); /* BLAS version */
      } 
      /* ... stabilized!! */
    } /* z update complete */
    

    /* calculate b[j]=||z||.... */
    for (xx=0.0,zp=z,p0=zp+*n;zp<p0;zp++) xx += *zp * *zp;b[j]=sqrt(xx); 
  
    /*if (b[j]==0.0&&j< *n-1) ErrorMessage(_("Lanczos failed"),1);*/ /* Actually this isn't really a failure => rank(A)<l+lm */
    /* get q[j+1]      */
    if (j < *n-1)
    { q[j+1]=(double *)calloc((size_t) *n,sizeof(double));
      for (xx=b[j],qp=q[j+1],p0=qp + *n,zp=z;qp<p0;qp++,zp++) *qp = *zp/xx; /* forming q[j+1]=z/b[j] */
    }

    /* Now get the spectral decomposition of T_j.  */

    if (((j>= *m + *lm)&&(j%f_check==0))||(j == *n-1))   /* no  point doing this too early or too often */
    { for (i=0;i<j+1;i++) d[i]=a[i]; /* copy leading diagonal of T_j */
      for (i=0;i<j;i++) g[i]=b[i]; /* copy sub/super diagonal of T_j */   
      /* set up storage for eigen vectors */
      if (vlength) free(v); /* free up first */
      vlength=j+1; 
      v = (double *)calloc((size_t)vlength*vlength,sizeof(double));
 
      /* obtain eigen values/vectors of T_j in O(j^2) flops */
    
      kk = j + 1;
      mgcv_trisymeig(d,g,v,&kk,1,1);
      /* ... eigenvectors stored one after another in v, d[i] are eigenvalues */

      /* Evaluate ||Tj|| .... */
      normTj=fabs(d[0]);if (fabs(d[j])>normTj) normTj=fabs(d[j]);

      for (k=0;k<j+1;k++) /* calculate error in each eigenvalue d[i] */
      { err[k]=b[j]*v[k * vlength + j]; /* bound on kth e.v. is b[j]* (jth element of kth eigenvector) */
        err[k]=fabs(err[k]);
      }
      /* and check for termination ..... */
      if (j >= *m + *lm)
      { max_err=normTj*eps_stop;
        if (biggest) { /* getting m largest magnitude eigen values */
	  /* only one convergence test is sane here:
             1. Find the *m largest magnitude elements of d. (*lm is 0)
             2. When all these have converged, we are done.
          */   
          pi=ni=0;converged=1;
          while (pi+ni < *m) if (fabs(d[pi])>= fabs(d[j-ni])) { /* include d[pi] in largest set */
              if (err[pi]>max_err) {converged=0;break;} else pi++;
	    } else { /* include d[j-ni] in largest set */
              if (err[ni]>max_err) {converged=0;break;} else ni++;
            }
   
          if (converged) {
            *m = pi;
            *lm = ni;
            j++;break;
          }
        } else /* number of largest and smallest supplied */
        { ok=1;
          for (i=0;i < *m;i++) if (err[i]>max_err) ok=0;
          for (i=j;i > j - *lm;i--) if (err[i]>max_err) ok=0;
          if (ok) 
          { j++;break;}
        }
      }
    }
  }
  /* At this stage, complete construction of the eigen vectors etc. */
  
  /* Do final polishing of Ritz vectors and load va and V..... */
  /*  for (k=0;k < *m;k++) // create any necessary new Ritz vectors 
  { va->V[k]=d[k];
    for (i=0;i<n;i++) 
    { V->M[i][k]=0.0; for (l=0;l<j;l++) V->M[i][k]+=q[l][i]*v[k][l];}
  }*/

  /* assumption that U is zero on entry! */

  for (k=0;k < *m;k++) /* create any necessary new Ritz vectors */
  { D[k]=d[k];
    for (l=0;l<j;l++)
    for (xx=v[l + k * vlength],p0=U + k * *n,p1 = p0 + *n,qp=q[l];p0<p1;p0++,qp++) *p0 += *qp * xx;
  }

  for (k= *m;k < *lm + *m;k++) /* create any necessary new Ritz vectors */
  { kk=j-(*lm + *m - k); /* index for d and v */
    D[k]=d[kk];
    for (l=0;l<j;l++)
    for (xx=v[l + kk * vlength],p0=U + k * *n,p1 = p0 + *n,qp=q[l];p0<p1;p0++,qp++) *p0 += *qp * xx;
  }
 
  /* clean up..... */
  free(a);
  free(b);
  free(g);
  free(d);
  free(z);
  free(err);
  if (vlength) free(v);
  for (i=0;i< *n+1;i++) if (q[i]) free(q[i]);free(q);  
  *n = j; /* number of iterations taken */
} /* end of Rlanczos */


double Bai(double *A,int *n,double *lambdamin, double *lambdamax, double *tol, double *rz, int *col) 
{
   int flag=1;
   double alpha=1;
   double beta=0;
   int incx=1;
   int dim;
   double *x_old, *x_old_old;
   double *v;
   double *e1, *e2;
   double sum_alpha;
   double sum_r2;
   const char uplo='U';
   int i,j;
   double bt,c1,c2;
   double *diag, *offdiag;
   double *r;
   double gamma_old=0;
   double *eigenvectors=NULL;
   int eigenvectors_length=0;
   double *g,*d;
   double sum;
   double Iold=0;
   int nrhs=1;
   int info=-1000;
   double *diag1, *diag2, *ud1, *ud2, *ld1,*ld2;
   double phi1, phi2;
   
   
   x_old=(double *)calloc((size_t) *n,sizeof(double));
   x_old_old=(double *)calloc((size_t) *n,sizeof(double));
   v=(double *)calloc((size_t) *n,sizeof(double));
   diag=(double *)calloc((size_t) *n,sizeof(double));
   offdiag=(double *)calloc((size_t) *n,sizeof(double));
   r=(double *)calloc((size_t) *n,sizeof(double));
   g=(double *)calloc((size_t) *n,sizeof(double));
   d=(double *)calloc((size_t) *n,sizeof(double));
   e1=(double *)calloc((size_t) *n,sizeof(double));
   e2=(double *)calloc((size_t) *n,sizeof(double));
   diag1=(double *)calloc((size_t) *n,sizeof(double));
   diag2=(double *)calloc((size_t) *n,sizeof(double));
   ud1=(double *)calloc((size_t) *n,sizeof(double));
   ud2=(double *)calloc((size_t) *n,sizeof(double));
   ld1=(double *)calloc((size_t) *n,sizeof(double));
   ld2=(double *)calloc((size_t) *n,sizeof(double));
   
   /*
   * Initial value for the random vector x_old, x_old_old, e1,e2
   */
   bt=sqrt(*n);
   c1=1.0/bt;
   c2=-1.0/bt;
   
   //GetRNGstate() and PutRNGstate() is not thread safe and cause a lot of problems with 
   //OpenMP
   
   //GetRNGstate(); 
   for (i=0;i < *n;i++)   
   {
     //Check if the uniform number is bigger than 0.5
     if(rz[i+(*col)*(*n)] > 0.5)
     {
       x_old[i]=c1;
     }else{
       x_old[i]=c2;
     }
     x_old_old[i]=0.0;
     e1[i]=0;
     e2[i]=0;
   }
   //PutRNGstate();
   
   j=-1;
   while(flag)
   {
      j++;
      /* Calculating v=A*x_old */
      F77_NAME(dsymv)(&uplo,n,&alpha,A,n,x_old,&incx,&beta,v,&incx); 
         
      /* Calculating alpha_j=v*x_old */
      sum_alpha=0;
      for(i=0; i<*n; i++) sum_alpha+=x_old[i]*v[i];
      diag[j]=sum_alpha;
      
      /* Calculating rj=v-alpha[j]*x_old-gamma_0*x_old_old */
      sum_r2=0;
      for(i=0; i<*n;i++){
	r[i]=v[i]-sum_alpha*x_old[i]-gamma_old*x_old_old[i];
	sum_r2+=r[i]*r[i];
      }
      offdiag[j]=sqrt(sum_r2);
      
      /* 
       * Updating x_old, x_old_old, gamma_old
       */
       for(i=0; i<*n;i++)
       {
	  x_old_old[i]=x_old[i];
	  x_old[i]=r[i]/offdiag[j];
       }
       gamma_old=offdiag[j];
       
       /*
	* Tridiagonal systems
	*/
       if(j>3)
       {
	  free(eigenvectors);
	  dim=j+1;
	  eigenvectors_length=dim*dim;
	  
	  eigenvectors = (double *)calloc((size_t)eigenvectors_length,sizeof(double));
	  
	  /*Copy elements*/
	  for (i=0;i<j+1;i++) {
	    d[i]=diag[i];
	    g[i]=offdiag[i];
	  }
	  
	  /* obtain eigen values/vectors of T_j in O(j^2) flops 
	  ... eigenvectors stored one after another in eigenvectors, d[i] are eigenvalues
	  Warning: dim is overwritten
	  */
	  
	  mgcv_trisymeig(d,g,eigenvectors,&dim,1,1);
	  dim=j+1;
	  
	  //Rprintf("j=%d\n",j);
	  
	  //Rprintf("Eigenvalues\n");
	  //for(i=0;i<j+1;i++)
	  //{
	  //  Rprintf("%f\n",d[i]);
	  //}
	  //Rprintf("Eigenvectors\n");
	  //for(i=0; i<eigenvectors_length;i++) Rprintf("%f\n",eigenvectors[i]);
	  //Rprintf("First elements:\n");
	  //for(i=0; i<dim;i++) Rprintf("%f\n",eigenvectors[dim*i]);
	  
	  
	  sum=0;
	  for(i=0;i<dim;i++)
	  {
	    sum=sum+eigenvectors[dim*i]*eigenvectors[dim*i]/d[i];
	  }
      
	  //Rprintf("Sum=%f\n",sum);
	  if(fabs(sum-Iold)<*tol*fabs(sum)) flag=0;
	  Iold=sum;
       }
   }
   
   //Rprintf("Iter=%d\n",j);
   
 /*
   *Ready for Gauss-Radau
   *Solve the tridiagonal systems, (T_j-aI)*delta=beta_j^2*e_j and (T_j-bI)*delta=beta_j^2*e_j 
   */
  
   e1[dim-1]=e2[dim-1]=offdiag[dim-1]*offdiag[dim-1];
   
   for(i=0; i<dim; i++)
   {
      diag1[i]=diag[i]-*lambdamin;
      diag2[i]=diag[i]-*lambdamax;
      ud1[i]=ud2[i]=offdiag[i];
   }
   
   F77_NAME(dgtsv)(&dim,&nrhs,ld1,diag1,ud1,e1,&dim,&info);
   
   if(info==0)
   {
      /*
      Rprintf("Successful solution of tridiagonal system\n");
      Rprintf("Solution:\n");
      for(i=0;i<dim;i++)
      {
        Rprintf("delta[%d]=%f\n",i,e1[i]);
      }
      */
    }else{
      Rprintf("dgtsv Error...code=%d\n",info);
    }
    
    phi1=*lambdamin+e1[dim-1];
    //Rprintf("phi1=%f\n",phi1);
    
    info=-1000;
    
    F77_NAME(dgtsv)(&dim,&nrhs,ld2,diag2,ud2,e2,&dim,&info);
    if(info==0)
    {
      /*
      Rprintf("Successful solution of tridiagonal system\n");
      Rprintf("Solution:\n");
      for(i=0;i<dim;i++)
      {
        Rprintf("delta[%d]=%f\n",i,e1[i]);
      }
      */
    }else{
      Rprintf("dgtsv Error...code=%d\n",info);
    }
    phi2=*lambdamax+e2[dim-1];
    //Rprintf("phi2=%f\n",phi2);
    
    /*
    Eigenvalues and eigensystems for T_(j+1).
    This is still part of Gauss-Radau
    */
    
    dim=j+1;
    for(i=0; i<dim; i++)
    {
      d[i]=diag[i];
      g[i]=offdiag[i];
    }
   
    d[dim]=phi1;
    free(eigenvectors);
    dim=j+2;
    eigenvectors_length=dim*dim;
    eigenvectors = (double *)calloc((size_t)eigenvectors_length,sizeof(double));
    mgcv_trisymeig(d,g,eigenvectors,&dim,1,1);
    dim=j+2;
    sum=0;
    for(i=0;i<dim;i++)
    {
	sum=sum+eigenvectors[dim*i]*eigenvectors[dim*i]/d[i];
    }
    
    //Rprintf("Sum=%f\n",*ans);
    
    dim=j+1;
    for(i=0; i<dim; i++)
    {
      d[i]=diag[i];
      g[i]=offdiag[i];
    }
   
    d[dim]=phi2;
    free(eigenvectors);
    dim=j+2;
    eigenvectors_length=dim*dim;
    eigenvectors = (double *)calloc((size_t)eigenvectors_length,sizeof(double));
    mgcv_trisymeig(d,g,eigenvectors,&dim,1,1);
    dim=j+2;
    
    for(i=0;i<dim;i++)
    {
	sum=sum+eigenvectors[dim*i]*eigenvectors[dim*i]/d[i];
    }
   
    
    free(x_old);
    free(x_old_old);
    free(v);
    free(diag);
    free(offdiag);
    free(g);
    free(d);
    free(e1);
    free(e2);
    free(diag1);
    free(diag2);
    free(ud1);
    free(ud2);
    free(ld1);
    free(ld2);  
    
    //Rprintf("Sum=%f\n",sum);
    return(*n * sum/2.0);
}
