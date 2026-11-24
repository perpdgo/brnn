# file brnn/adam.R
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 or 3 of the License
#  (at your option).
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  A copy of the GNU General Public License is available at
#  http://www.r-project.org/Licenses/

#A package for Bayesian Regularized Neural Networks
#Author: Paulino Perez Rodriguez
#Madison, WI, Sep. 2012
#Birmingaham, Alabama, Jan. 2013
#East Lansing, Michigan, Jan. 2020
#Texcoco, Mexico, Nov. 2023

#WARNING: This is an experimental version

# ADAM algorithm

##################################################################################################


adam.brnn.default=function(x,y,neurons=2,lambda=NULL,alpha=0.05,beta1=0.9,beta2=0.999,
                           eps=1e-8,change=0.001,min_grad=1e-10,epochs=500,normalize=TRUE,
                           theta_start=NULL,K=10,start.method="Xavier-Glorot",verbose=FALSE,
                           WarmStart=TRUE,lambda.criterion="lambda.min",
                           lambda_par=list(lambda_max=10,lambda_min=0.0001,nlambda=10))
{
  reason="UNKNOWN";
  
  #Checking that the inputs are ok
  if(!is.vector(y)) stop("y must be a vector\n")
  if(!is.matrix(x)) x=as.matrix(x)
  
  n=nrow(x)
  p=ncol(x)
  
  #lambda
  if(is.null(lambda)|(is.numeric(lambda)&length(lambda)>1))
  {
    lambda=cv.adam.brnn(x=x,y=y,neurons=neurons,K=K,WarmStart=WarmStart,lambda_seq=lambda,
                        alpha=alpha,beta1=beta1,beta2=beta2,eps=eps,change=change,
                        epochs=epochs,normalize=normalize,start.method=start.method,
                        lambda_par=lambda_par,lambda.criterion=lambda.criterion,verbose=verbose)
  }else if(!is.numeric(lambda)|length(lambda)==0){
    stop("A vector or a numerical value is required for the lambda parameter.")
  }
  
  if(normalize)
  {
    x_base=apply(x,2,min)
    x_spread=apply(x,2,max)-x_base 
    x_normalized=normalize(x,base=x_base,spread=x_spread)
    
    y_base=min(y)
    y_spread=max(y)-y_base
    y_normalized=normalize(y,base=y_base,spread=y_spread)
  }else{
    y_normalized=y
    y_base=NULL
    y_spread=NULL
    
    x_normalized=x
    x_base=NULL
    x_spread=NULL
  }
  
  #Initializing parameters for the net
  npar=neurons*(1+1+p)
  if(verbose) cat("Number of parameters (weights and biases) to estimate:",npar,"\n")
  
  size_bloq=p+2
  
  if(is.null(theta_start))
  {
    if(start.method=="Nguyen-Widrow")
    {
      if(!verbose)
      {
        capture.output(theta_start<-initnw(neurons=neurons,p=p,n=n,npar=npar))
      }else{
        theta_start<-initnw(neurons=neurons,p=p,n=n,npar=npar)
      }
      theta=matrix(unlist(theta_start),nrow=neurons,ncol=size_bloq,byrow=TRUE)
    }else{
      theta=initnw_alt(neurons=neurons,p=p)
    }
  }else{
    theta=matrix(unlist(theta_start),nrow=neurons,ncol=size_bloq,byrow=TRUE)
  }
  
  m=matrix(0,nrow=neurons,ncol=size_bloq)
  v=matrix(0,nrow=neurons,ncol=size_bloq)
  g=matrix(0,nrow=neurons,ncol=size_bloq)
  unos=rep(1,n)
  
  epoch=1
  flag_gradient=TRUE
  flag_change_F=TRUE
  flag_change_Ed=TRUE
  F_history=numeric()
  
  while(epoch<=epochs & flag_gradient & flag_change_Ed & flag_change_F)
  {
    if(verbose)
    {
      cat("----------------------------------------------------------------------\n")
      cat("Epoch=",epoch,"\n")
    }
    
    w=theta[,1]
    b=theta[,2]
    beta_p=theta[,3:size_bloq,drop=FALSE]
    
    u=base::tcrossprod(beta_p,x_normalized)+b
    g_u=tanh(u)
    pred=c(base::crossprod(w,g_u))
    
    e=y_normalized-pred
    Ed=sum(e^2)
    Ew=sum(w^2)+sum(beta_p^2)
    C=Ed+lambda*Ew
    
    F_history[epoch]=C
    
    comun=-2*e
    g_w=c(g_u%*%comun)+(2*lambda*w)
    delta=(base::tcrossprod(w,comun))*(1-g_u^2)
    g_b=c(delta%*%unos)
    g_beta=delta%*%x_normalized+(2*lambda*beta_p)
    
    g[,1]=g_w
    g[,2]=g_b
    g[,3:size_bloq]=g_beta
    
    mg=max(abs(g))
    flag_gradient=mg>min_grad
    
    if(verbose)
    {
      cat("C=",C,"\tEd=",Ed,"\tEw=",Ew,"\n")
      cat("gradient=",mg,"\n")
    }
    
    if(epoch>3)
    {
      if(max(abs(diff(F_history[(epoch-3):epoch])))<change)
      {
        flag_change_F=FALSE;
        reason=paste("Changes in F= SCE + lambda*Ew in last 3 iterations less than",change,sep=" ");
      }
    }
    
    if(Ed<0.01)
    {
      flag_change_Ed=FALSE
      reason="SCE <= 0.01";
    }
    
    if(flag_gradient & flag_change_Ed & flag_change_F)
    {
      m=beta1*m+(1-beta1)*g
      v=beta2*v+(1-beta2)*(g^2)
      
      m_hat=m/(1-beta1^epoch)
      v_hat=v/(1-beta2^epoch)
      
      theta=theta-alpha*m_hat/(sqrt(v_hat)+eps)
      epoch=epoch+1
    }
  }
  
  if((epoch-1)==epochs) reason="Maximum number of epochs reached";
  if(!flag_gradient) reason="Minimum gradient reached";
  
  theta=unname(split(theta,rep(1:neurons,size_bloq)))
  
  #answer
  out=list(theta=theta,lambda=lambda,Ed=Ed,Ew=Ew,F_history=F_history,reason=reason,epoch=epoch,
           neurons=neurons,p=p,n=n,npar=npar,x_normalized=x_normalized,x_base=x_base,x_spread=x_spread,
           y_base=y_base,y_spread=y_spread,y=y,
           normalize=normalize)
  out$call=match.call()
  class(out)=c("adam.brnn","brnn");
  
  #return the goodies
  return(out)
}



#K-fold cross-validation for selecting lambda in adam
cv.adam.brnn=function(x,y,neurons,K,WarmStart=TRUE,lambda_seq=NULL,alpha=0.05,beta1=0.9,beta2=0.999,
                      eps=1e-8,change=0.001,epochs=500,normalize=TRUE,start.method="Xavier-Glorot",
                      lambda_par=list(lambda_max=10,lambda_min=0.001,nlambda=10),
                      lambda.criterion="lambda.min",verbose=FALSE)
{
  if(!is.vector(y)) stop("y must be a vector\n")
  X=as.matrix(x)
  n=nrow(X)
  p=ncol(X)
  if(K<2|K>n) stop("K must be an integer between 2 and nrow(x)\n")
  
  size_bloq=p+2
  folds=sample(rep(1:K,length.out=n))
  
  #lambda sequence (decreasing for warm start)
  if(is.null(lambda_seq))
  {
    lambda_seq=get.lambda.seq(lambda_max=lambda_par$lambda_max,
                              lambda_min=lambda_par$lambda_min,
                              nlambda=lambda_par$nlambda)
  }
  lambda_seq=sort(lambda_seq,decreasing=TRUE)
  n_lambda=length(lambda_seq)
  
  matriz_res=matrix(NA_real_,nrow=n_lambda,ncol=K)
  
  #cross-validation
  for(fold in 1:K)
  {
    if(verbose) cat("\nFold =",fold,"\n")
    
    indices_test=which(folds==fold)
    n_test=length(indices_test)
    
    X_train=X[-indices_test,,drop=FALSE]
    y_train=y[-indices_test]
    X_test=X[indices_test,,drop=FALSE]
    y_test=y[indices_test]
    
    #normalize just once
    if(normalize)
    {
      x_base=apply(X_train,2,min)
      x_spread=apply(X_train,2,max)-x_base
      y_base=min(y_train)
      y_spread=max(y_train)-y_base
      
      X_train=normalize(X_train,base=x_base,spread=x_spread)
      X_test=normalize(X_test,base=x_base,spread=x_spread)
      y_train=normalize(y_train,base=y_base,spread=y_spread)
      y_test=normalize(y_test,base=y_base,spread=y_spread)
      
      #un_normalize is lineal: y-yhat=(spread/2)*(y_norm-yhat_norm)
      escala=(y_spread/2)^2
    }else{
      escala=1
    }
    
    theta_ini=NULL
    
    for(l_idx in 1:n_lambda)
    {
      lambda=lambda_seq[l_idx]
      
      if(verbose) cat("  Lambda =",lambda,"\n")
      
      #adam
      res=adam.brnn(x=X_train,y=y_train,neurons=neurons,lambda=lambda,alpha=alpha,beta1=beta1,
                    beta2=beta2,eps=eps,change=change,epochs=epochs,normalize=FALSE,
                    theta_start=theta_ini,start.method=start.method,verbose=FALSE)
      
      if(WarmStart) theta_ini=res$theta
      
      #prediction in test (S x n_test)
      theta=matrix(unlist(res$theta),nrow=neurons,ncol=size_bloq,byrow=TRUE)
      u_test=base::tcrossprod(theta[,3:size_bloq,drop=FALSE],X_test)+theta[,2]
      pred=c(base::crossprod(theta[,1],base::tanh(u_test)))
      
      #compute directly unnormalized
      matriz_res[l_idx,fold]=escala*sum((y_test-pred)^2)/n_test
    }
  }
  
  # we use the matrix approach
  cvm=drop(matriz_res %*% rep(1 / K, K))
  cvsd=sqrt(pmax(0, (drop((matriz_res^2) %*% rep(1, K)) - K * (cvm^2)) / (K - 1)))
  lambda_opt=getopt.lambda(lambda=lambda_seq,
                           cvm=cvm,
                           cvsd=cvsd,
                           lambda.criterion=lambda.criterion)
  
  return(lambda_opt)
}



# Xavier/Glorot method
initnw_alt = function(neurons,p){
  
  # cat("Xavier/Glorot method\nS")
  
  lim_beta=sqrt(6 / (p + neurons))
  lim_w=sqrt(6 / (neurons + 1)) # una salida
  
  theta_matrix=matrix(0, nrow = neurons, ncol = p + 2)
  
  # b
  theta_matrix[, 1]=runif(neurons, -lim_w, lim_w) 
  
  # w
  theta_matrix[, 2]=runif(neurons, -lim_w, lim_w)
  
  # betas
  theta_matrix[, 3:(p + 2)]=runif(neurons * p, -lim_beta, lim_beta)
  
  return(theta_matrix)
}

# lambda sequence
get.lambda.seq = function(lambda_max = 10, lambda_min = 0.001, nlambda = 10) {
  lambda_seq = 10^seq(from=log10(lambda_max),to=log10(lambda_min),length.out=nlambda)
  return(lambda_seq)
}


# get lambda optimized
getopt.lambda=function(lambda,cvm,cvsd, lambda.criterion = "lambda.1se"){
  idmin = which.min(cvm)
  lambda.min = lambda[idmin]
  
  semin = cvm[idmin] + cvsd[idmin]
  id1se = which(cvm <= semin)
  lambda.1se = max(lambda[id1se])
  
  lambda_final = if(lambda.criterion == "lambda.1se"){
    lambda.1se
  }else{
    lambda.min
  }
  return(lambda_final)
}

# aditional
adam.brnn = function(x, ...) UseMethod("adam.brnn")

adam.brnn.formula = function(formula, data, contrasts = NULL, ...) {
  m=match.call(expand.dots = FALSE)
  if(is.matrix(eval.parent(m$data))) m$data=as.data.frame(data)
  m$... = m$contrasts = NULL
  m[[1L]] = as.name("model.frame")
  m = eval.parent(m)
  Terms = attr(m, "terms")
  x =model.matrix(Terms, m, contrasts)
  cons = attr(x, "contrast")
  xint = match("(Intercept)", colnames(x), nomatch=0L)
  if(xint > 0L) x = x[, -xint, drop=FALSE]
  
  y = model.response(m)
  
  out = adam.brnn.default(x, y, ...)
  
  out$terms = Terms
  out$coefnames = colnames(x)
  out$call = match.call()
  out$contrasts = cons
  out$xlevels = .getXlevels(Terms, m)
  
  class(out) = c("adam.brnn.formula", "adam.brnn", "brnn")
  
  return(out)	
}

##################################################################################################
