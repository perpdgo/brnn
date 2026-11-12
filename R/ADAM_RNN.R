
rm(list=ls())


library(dplyr)
# library(glmnet)
library(brnn)
# data(QuickStartExample)

# dyn.load("./win/util_win.dll")
# 
# #funciones del brnn
# #---------------------------------
# 
# #This function will obtain predicted values for a basic NN
# #It uses an omp version of the routine predictions.nn to do the work
# #This is an internal function
# predictions.nn.C=function(vecX,n,p,theta,neurons,cores=1)
# {
#   yhat=rep(NA,n)
#   vectheta=as.vector(unlist(theta))
#   out=.Call("predictions_nn",as.double(vecX), as.integer(n), 
#             as.integer(p), as.double(vectheta), 
#             as.integer(neurons),
#             as.double(yhat),
#             as.integer(cores))
#   yhat=out[[1]]
#   return(yhat)
# }
# 
# normalize=function(x,base,spread)
# {
#   if(is.matrix(x))
#   {
#     return(sweep(sweep(x,2,base,"-"),2,2/spread,"*")-1)
#   }else{
#     return(2*(x-base)/spread-1)
#   }
# }
# 
# #Go back to the original scale
# #x=base+0.5*spread*(z+1)
# un_normalize=function(z,base,spread)
# {
#   if(is.matrix(z))
#   {
#     return(sweep(sweep(z+1,2,0.5*spread,"*"),2,base,"+"))
#   }else{
#     return(base+0.5*spread*(z+1))
#   }	
# }
# 
# 
# 
# #tanh function
# tanh=function(x)
# {
#   (exp(2*x)-1)/(exp(2*x)+1)
# }
# 
# #tansig function, the evaluation of this function 
# #is faster than tanh and the derivative is the same!!!
# tansig=function(x) 2/(1+exp(-2*x)) - 1 
# 
# #sech function
# sech=function(x)
# {
#   2*exp(x)/(exp(2*x)+1)
# }
# 
# 
# initnw=function(neurons,p,n,npar)
# {
#   theta=list()
#   for(i in 1:neurons)
#   {
#     theta[[i]]=runif(npar/neurons,-0.5,0.5)
#   }
#   
#   cat("Nguyen-Widrow method\n")
#   
#   if(p==1)
#   {
#     scaling_factor=0.7*neurons;
#     cat("Scaling factor=",scaling_factor,"\n")
#     for(i in 1:neurons)
#     {
#       lambda=theta[[i]]
#       weight=lambda[1]
#       bias=lambda[2]
#       lambda=lambda[-c(1,2)]
#       lambda=scaling_factor
#       bias=runif(1,-scaling_factor,scaling_factor)
#       theta[[i]]=c(weight,bias,lambda)
#     }
#   }else{
#     scaling_factor=0.7*(neurons)^(1.0/n)
#     cat("Scaling factor=",scaling_factor,"\n")
#     b=seq(-1,1,length.out=neurons)
#     for(i in 1:neurons)
#     {
#       lambda=theta[[i]]
#       weight=lambda[1]
#       bias=lambda[2]
#       lambda=lambda[-c(1,2)]         
#       norm=sqrt(sum(lambda^2))
#       lambda=scaling_factor*lambda/norm
#       bias=scaling_factor*b[i]*sign(lambda[1])
#       theta[[i]]=c(weight,bias,lambda)
#     }
#   }   
#   return(theta)
# }

# esta es una propuesta:
# Xavier/Glorot
initnw_alt = function(neurons,p){
  
  # cat("Xavier/Glorot method\n")
  
  lim_beta=sqrt(6 / (p + neurons))
  lim_w=sqrt(6 / (neurons + 1)) # una salida
  
  theta_matrix=matrix(0, nrow = neurons, ncol = p + 2)
  colnames(theta_matrix)=c("b", "w", paste0("beta_", 1:p))
  
  #inciamos
  # b
  theta_matrix[, 1]=runif(neurons, -lim_w, lim_w) 
  
  # w
  theta_matrix[, 2]=runif(neurons, -lim_w, lim_w)
  
  # betas
  theta_matrix[, 3:(p + 2)]=runif(neurons * p, -lim_beta, lim_beta)
  
  return(theta_matrix)
}


#---------------------------------

#tercer intento, vamos a normalizar y desnormalizar
# iniciamos un index para cara vbloque de parámetros
#para seguir usando la facilidad de R al manejar todo vectorialmente

# ADAM
# ADAM con el bloque de normalización dentro
adam_nn.default = function(x, y, S=2, lambda=NULL, alpha=0.05, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-6, max_iter=1000, normalize=TRUE, theta_start = NULL,
                   K=10,start.method="Xavier-Glorot",verbose = FALSE,WarmStart=TRUE,criterio_lambda = "lambda.min",lambda_par=list(lambda_max = 10,lambda_min = 0.0001,nlambda = 10)){
  x = as.matrix(x)
  n = nrow(x)
  p = ncol(x)
  
  #revisamos si hay un lambda
  if(is.null(lambda) | (is.numeric(lambda) & length(lambda) > 1) ){
    lambda = cv_adam_nn(x = x,y = y,S = S,K=K,WarmStart = WarmStart,lambda_seq = lambda,
                        alpha=alpha, beta1=beta1, beta2=beta2, eps=eps,tol=tol,
                        max_iter=max_iter, normalize=normalize, start.method=start.method,
                        lambda_par = lambda_par, criterio_lambda = criterio_lambda, verbose = verbose)
  }else if (!is.numeric(lambda) | length(lambda) == 0){
    stop("Se requiere un vector o un valor numérico para el parámetro lambda.") # inglés
  }
  
  
  # bloque de normalización copiado del brnn :D
  if(normalize)
  {
    x_base = apply(x, 2, min)
    x_spread = apply(x, 2, max) - x_base 
    x_normalized = brnn::normalize(x, base=x_base, spread=x_spread)
    
    y_base = min(y)
    y_spread = max(y) - y_base
    y_normalized = brnn::normalize(y, base=y_base, spread=y_spread)
    
  }else{
    y_normalized = y
    y_base = NULL
    y_spread = NULL
    
    x_normalized = x
    x_base = NULL
    x_spread = NULL
  }
  
  # Dimensión: S para w, S para b, S*p para beta
  dim_theta = S + S + S*p
  
  tam_bloque = p + 2
  
  if (is.null(theta_start)) {
    
    if(start.method == "Nguyen-Widrow"){
      if (!verbose) {
        capture.output(theta_start<-initnw(neurons = S, p = p, n = n, npar = dim_theta))
      }
      theta = matrix(unlist(theta_start), nrow = S, ncol = tam_bloque, byrow = TRUE)
    }else{
      theta=initnw_alt(neurons = S, p = p)
    }
    
  } else {
    theta = matrix(unlist(theta_start), nrow = S, ncol = tam_bloque, byrow = TRUE)
  }
  
  m = matrix(0, nrow = S, ncol = tam_bloque)
  v = matrix(0, nrow = S, ncol = tam_bloque)
  g = matrix(0, nrow = S, ncol = tam_bloque)
  t_i = 0
  sce_ant = Inf
  unos=rep(1, nrow(x_normalized))
  
  # t_x_normalized = t(x_normalized)
  # ADAM (pasos)
  for(step in 1:max_iter){
    t_i = t_i + 1

    w = theta[,1]
    b = theta[,2]
    beta_p = theta[,3:tam_bloque]
    
    #FIXME: #cross product o t-cross product
    u = base::tcrossprod(beta_p, x_normalized) + b #beta %*% t_x_normalized + b
    g_u = tanh(u) # variable
    pred = c(base::crossprod(w, g_u))#as.vector(t(w) %*% g_u)
    
    # residuales
    residual = y_normalized - pred
    
    # criterio de parada si el SCE ya no cambia
    sce_act = sum(residual^2) + lambda * (sum(w^2) + sum(beta_p^2))
    if(abs(sce_ant-sce_act) < tol){
      t_i = t_i - 1
      break
    }
    sce_ant = sce_act
    comun = -2 * residual
    
    g_w = c(g_u %*% comun) + (2 * lambda * w)
    delta =  (base::tcrossprod(w,comun)) * (1 - g_u^2)
    g_b = c(delta %*% unos)
    g_beta = delta %*% x_normalized + (2 * lambda * beta_p)
    
    g[,1] = g_w
    g[,2] = g_b
    g[,3:tam_bloque] = g_beta
    
    # iteramos el momento 1
    m = beta1 * m + (1 - beta1) * g
    # iteramos el momento 2
    v = beta2 * v + (1 - beta2) * (g^2)

    # iteramos con los sesgos
    m_hat = m / (1 - beta1^t_i)
    v_hat = v / (1 - beta2^t_i)

    theta = theta - alpha * m_hat / (sqrt(v_hat) + eps)
    
    # # calculamos el paso ajustado por sesgo (escalar)
    # alpha_t = alpha * sqrt(1 - beta2^t_i) / (1 - beta1^t_i)
    # 
    # # actualizamos parámetros directamente sin crear m_hat ni v_hat
    # theta = theta - alpha_t * m / (sqrt(v) + eps)

  }
  
  theta = unname(split(theta, rep(1:S,tam_bloque)))
  
  res_final = list(
    theta = theta,
    neurons = S,
    p = p,
    n = n,
    normalize = normalize,
    lambda = lambda,
    iteraciones = t_i,
    sce_final = sum((y_normalized - pred)^2),
    x_normalized = x_normalized,
    x_base = x_base,
    x_spread = x_spread,
    y_base = y_base,
    y_spread = y_spread
  )
  
  class(res_final) = c("adam_nn", "brnn")
  return(res_final)
  
}


# FIXME: ver la función de predecir de brnn (revisar para usarlo)
#función para calcular los valores predichos usando los parámetros estimados
predict_nn = function(X, theta, S, x_base = NULL, x_spread = NULL, y_base = NULL, y_spread = NULL) {
  X = as.matrix(X)
  p = ncol(X) #revisamos la dimensión de X
  
  theta = unlist(theta) # lo utilizo en lo que reviso el prdict del brnn
  
  # Si se proporcionó la base de X, primero normalizamos los datos de entrada
  if(!is.null(x_base) & !is.null(x_spread)){
    X = brnn::normalize(X, base = x_base, spread = x_spread)
  }
  
  # Dimensión total del vector de parámetros theta: S para w, S para b, S*p para beta
  dim_theta = S + S + S*p
  
  # Cada bloque de neurona ocupa (p + 2) posiciones consecutivas en theta
  tam_bloque = p + 2
  
  w_idx = seq(1, dim_theta, by = tam_bloque)
  b_idx = seq(2, dim_theta, by = tam_bloque)
  
  # Para las betas, generamos las p posiciones siguientes al sesgo en cada bloque, por complemento
  beta_idx = (1:dim_theta)[-c(w_idx, b_idx)]
  
  #extraemos los parámetros de theta manteniendo el orden indexado
  w = theta[w_idx]
  b = theta[b_idx]
  beta = matrix(theta[beta_idx], nrow=S, ncol=p, byrow=TRUE)
  
  # calculamos la activación de la capa oculta
  u = beta %*% t(X) + b #FIXME: X %*% beta + b #ver qupe sale y convertir: as.vector o as.numeric()
  g_u = tanh(u)
  
  #generamos la predicción final
  pred = as.vector(t(w) %*% g_u)
  
  # Si se proporcionaron los parámetros de escala, regresamos a la escala original
  if(!is.null(y_base) & !is.null(y_spread)){
    pred = brnn::un_normalize(pred, base = y_base, spread = y_spread)
  }
  
  return(pred)
}

#-----------------------------------------------------------------------------------------------
# aqui hacemos algo parecido al glmnet

# Función: calcular_lambda_seq
# adecuado de glmnet.R
# --------------------------------------------------------------------------------------------
calcular_lambda_seq = function(lambda_max = 10, lambda_min = 0.001, nlambda = 10) {
  # Generamos una progresión geométrica decreciente (escala logarítmica base 10)
  # rquivale a usar exp(seq(log(max), log(min), length.out = nlambda))
  lambda_seq = 10^seq(from=log10(lambda_max),to=log10(lambda_min),length.out=nlambda)
  return(lambda_seq)
}

#-----------------------------------------------------------------------------------------------

#caso del warm start
# línea 158 "Supply instead a decreasing sequence of lambda values. glmnet relies on its warm starts for speed..."
# A un lambda muy alto (10.0), la solución es trivial (casi todos los coeficientes son cero). El modelo converge rápido.
# Luego, usa esos números como warm start para el siguiente lambda un poco más pequeño.
# El "camino de regularización" (regularization path) se descubre soltando la penalización poco a poco.
# de mayor a menor

cv_adam_nn = function(x,y,S,K,WarmStart=TRUE,lambda_seq = NULL,alpha=0.05, beta1=0.9, beta2=0.999,
                      eps=1e-8, tol=1e-6, max_iter=1000, normalize=TRUE,start.method="Xavier-Glorot",
                         lambda_par=list(lambda_max = 10,lambda_min = 0.001,nlambda = 10),
                      criterio_lambda = "lambda.min",verbose = FALSE){
  X=as.matrix(x)
  n=nrow(X)
  folds=sample(rep(1:K,length.out=n))

  if (is.null(lambda_seq)) {
    lambda_seq = calcular_lambda_seq(lambda_max = lambda_par$lambda_max,
                                     lambda_min = lambda_par$lambda_min,
                                     nlambda = lambda_par$nlambda)
  }else{
    lambda_seq = sort(lambda_seq, decreasing = TRUE)
  }
  
  n_lambda = length(lambda_seq)

  matriz_res = matrix(NA, nrow = n_lambda, ncol = K)
  
  #cross-validation
  for(fold in 1:K){
    
    if(verbose){
      cat("\nFold =",fold,"\n")
    }
    
    indices_test=which(folds==fold)
    indices_train=which(folds!=fold)
    
    # base de entrenamiento
    X_train=X[indices_train,,drop=FALSE]
    y_train=y[indices_train]
    
    # base de validación
    X_test=X[indices_test,,drop=FALSE]
    y_test=y[indices_test]
    
    initial_theta = NULL
    
    for(l_idx in 1:n_lambda){
      
      lambda = lambda_seq[l_idx]
      
      if(verbose){
        cat("  Lambda =",lambda,"\n") #imprimo para saber donde voy por lambda
      }
      
      # adam
      res=adam_nn(x=X_train,y=y_train,lambda=lambda,S=S,theta_start = initial_theta,
                  alpha=alpha, beta1=beta1, beta2=beta2, eps=eps,tol=tol,
                  max_iter=max_iter, normalize=normalize,start.method=start.method)
      
      if(WarmStart){
        initial_theta = res$theta
      }
      # agregamos para ejecutarse una vez por fold --------------------------- Ojo aquí :D
      if (l_idx == 1) {
        X_test_norm = brnn:::normalize(X_test, base = res$x_base, spread = res$x_spread)
      }
      
      theta = matrix(unlist(res$theta), nrow = res$neurons, ncol = res$p+2, byrow = TRUE)
      u_test = base::tcrossprod(theta[,3:(res$p+2),drop = FALSE], X_test_norm) + theta[,2]
      pred = c(base::crossprod(theta[,1], tanh(u_test)))
      # w = theta[,1]
      # b = theta[,2]
      # beta_p = theta[,3:tam_bloque]
      
      pred = brnn:::un_normalize(pred, res$y_base, res$y_spread)
      
      # pred = predict(res,X_test)
      
      residual=y_test-pred
      SCE=sum(residual^2)
      
      matriz_res[l_idx, fold] = SCE
    }
  }
  
  lambda_opt=getopt.lambda(lambda=lambda_seq,
                           cvm=apply(matriz_res,1,mean),
                           cvsd=apply(matriz_res,1,sd),
                           criterio_final = criterio_lambda)
  
  return(lambda_opt)
}


getopt.lambda=function(lambda,cvm,cvsd, criterio_final = "lambda.1se"){
  # extraemos lambdas Óptimos (lógica igual a getOptcv.glmnet.R)
  idmin = which.min(cvm)
  lambda.min = lambda[idmin]#lambda con el mínimo error absoluto
  
  semin = cvm[idmin] + cvsd[idmin]#ponemos un umbral de tolerancia (1 Error Estándar)
  id1se = which(cvm <= semin) #+ 1e-11 #podemos sumar algo para evitar problemas
  lambda.1se = max(lambda[id1se]) #el lambda más restrictivo bajo el umbral
  
  # aquí voy a hacer el ajuste usando el criterio del mejor ajuste
  lambda_final = if(criterio_final == "lambda.1se"){
    lambda.1se
  }else{
    lambda.min
  }
  return(lambda_final)
}

# 
# esto lo copié del brnn:
adam_nn = function(x, ...) UseMethod("adam_nn")

# lo copié del brnn, pero lo llamé adam_nn, porque así corre mi función
adam_nn.formula = function(formula, data, contrasts = NULL, ...) {
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
  
  #FIXME: Aquí lo cambié, pero debo preguntar
  out = adam_nn.default(x, y, ...)
  
  # igualito:
  out$terms = Terms
  out$coefnames = colnames(x)
  out$call = match.call()
  out$contrasts = cons
  out$xlevels = .getXlevels(Terms, m)
  
  class(out) = c("adam_nn.formula", "adam_nn", "brnn")
  
  return(out)	
}

#------------
# ahora lo hago con cross-validation
# set.seed(123)
K=10 #de ejemplo usamos 10 grupos
S=2
#Generating the data
x1=seq(0,0.23,length.out=25)
y1=4*x1+rnorm(25,sd=0.1)
x2=seq(0.25,0.75,length.out=50)
y2=2-4*x2+rnorm(50,sd=0.1)
x3=seq(0.77,1,length.out=25)
y3=4*x3-4+rnorm(25,sd=0.1)
X=c(x1,x2,x3) #%>% as.data.frame()
y=c(y1,y2,y3)

#------------------------------------------------

out=brnn(y~X,neurons=2)
(out$alpha / out$beta)


res = adam_nn(x=X,y=y,S=S)#,lambda_par = list(lambda_max = 0.02, lambda_min=0.00001,nlambda=20))
# res = adam_nn(y~X,S=S)
res$lambda

res$theta

plot(X,y,xlim=c(0,1),ylim=c(-1.5,1.5),
     main="Bayesian Regularization for ANN 1-2-1")
lines(X,predict(res),
      col="blue",lty=2)


plot(predict(out) ~ predict(res),pch = 19,
     xlab = "adam",ylab = "brnn"
)
abline(0,1,col="red")


# primera prueba de eficiencia
Rprof("perfil.out")
res_cv=adam_nn(x=X, y=y, S = S)
Rprof(NULL)
summaryRprof("perfil.out")$by.self

# usando profvis, es más interactivo
library(profvis)
profvis({
  res_cv <- adam_nn(x = X, y = y, S = S)
})


# 
###############################################################
#Example 2
#sin wave function, example in the Matlab 2010b demo.

x = seq(-1,0.5,length.out=100)
y = sin(2*pi*x)+rnorm(length(x),sd=0.1)

#With the formula interface
out=brnn(y~x,neurons=3)
(out$alpha / out$beta)
# 

# comparamos con la red con ADAM:
# usamos las mismas neuronas: 3
S = 3
# set.seed(123)

res = adam_nn(x=x,y=y,S=S)#,lambda_par = list(lambda_max = 0.02, lambda_min=0.00001,nlambda=20))

res$lambda

plot(x,y)
lines(x,predict(res),
      col="blue",lty=2)
legend("topright",legend="Fitted model",col="blue",lty=2,bty="n")


plot(predict(out) ~ predict(res),pch = 19,
     xlab = "adam",ylab = "brnn"
)
abline(0,1,col="red")

# prueba de eficiencia
Rprof("perfil.out")
res_cv=adam_nn(x=x, y=y, S = S)
Rprof(NULL)
summaryRprof("perfil.out")$by.self


# hasta aquí el segundo ejemplo

###############################################################
#Example 3
#2 Inputs and 1 output
#the data used in Paciorek and
#Schervish (2004). The data is from a two input one output function with Gaussian noise
#with mean zero and standard deviation 0.25

data(twoinput)

#Formula interface
out=brnn(y~x1+x2,data=twoinput,neurons=10)



#With the default S3 method
#out=brnn(y=as.vector(twoinput$y),x=as.matrix(cbind(twoinput$x1,twoinput$x2)),neurons=10)

f=function(x1,x2) predict(out,cbind(x1,x2))
x1=seq(min(twoinput$x1),max(twoinput$x1),length.out=50)
x2=seq(min(twoinput$x2),max(twoinput$x2),length.out=50)
z=outer(x1,x2,f) # calculating the density values

transformation_matrix=persp(x1, x2, z,
                            main="Fitted model",
                            sub=expression(y==italic(g)~(bold(x))+e),
                            col="lightgreen",theta=30, phi=20,r=50, 
                            d=0.1,expand=0.5,ltheta=90, lphi=180,
                            shade=0.75, ticktype="detailed",nticks=5)
points(trans3d(twoinput$x1,twoinput$x2, f(twoinput$x1,twoinput$x2), 
               transformation_matrix), col = "red")

###############################################################
# Con ADAM

#Formula interface
S=10
# set.seed(123)
# res=adam_nn(x=twoinput[,c("x1","x2")],y=twoinput$y,S=S,max_iter = 1000)

res = adam_nn(x=twoinput[,c("x1","x2")],y=twoinput$y,S=S)


#With the default S3 method
#out=brnn(y=as.vector(twoinput$y),x=as.matrix(cbind(twoinput$x1,twoinput$x2)),neurons=10)

f=function(x1,x2)  predict(res,cbind(x1,x2))
x1=seq(min(twoinput$x1),max(twoinput$x1),length.out=50)
x2=seq(min(twoinput$x2),max(twoinput$x2),length.out=50)
z=outer(x1,x2,f) # calculating the density values

transformation_matrix=persp(x1, x2, z,
                            main="Fitted model",
                            sub=expression(y==italic(g)~(bold(x))+e),
                            col="lightgreen",theta=30, phi=20,r=50,
                            d=0.1,expand=0.5,ltheta=90, lphi=180,
                            shade=0.75, ticktype="detailed",nticks=5)
points(trans3d(twoinput$x1,twoinput$x2, f(twoinput$x1,twoinput$x2),
               transformation_matrix), col = "red")



plot(predict(out) ~ predict(res),pch = 19,
     xlab = "adam",ylab = "brnn"
     )
abline(0,1,col="red")


# prueba de eficiencia
Rprof("perfil.out")
res_cv=adam_nn(x=twoinput[,c("x1","x2")],y=twoinput$y,S=S)
Rprof(NULL)
summaryRprof("perfil.out")$by.self


# ----------------------------------------------------------------------
#Example 4
#Gianola et al. (2011).
#Warning, it will take a while

#Load the Jersey dataset
data(Jersey)

#Fit the model with the FULL DATA
#Formula interface
tiempo=data.frame(metodo=c("brnn","adam"),inicio=as.POSIXct(rep(NA, 2)),fin=as.POSIXct(rep(NA, 2)))

tiempo$inicio[1]=Sys.time()

out=brnn(pheno$yield_devMilk~G,neurons=2)

tiempo$fin[1]=Sys.time()

(out$alpha / out$beta)

#Obtain predictions and plot them against fitted values
plot(pheno$yield_devMilk,predict(out))

# Hacemos la prueba con ADAM --------------------------
tiempo$inicio[2]=Sys.time()
res = adam_nn(pheno$yield_devMilk~G,alpha=0.0001,S=2,max_iter=10000)#,lambda=0.1) #,start.method = "Nguyen-Widrow"
res = adam_nn(pheno$yield_devMilk~G,alpha=0.0001,S=2,max_iter=10000,lambda=0.5)
tiempo$fin[2]=Sys.time()

res$lambda

#Obtain predictions and plot them against fitted values
plot(pheno$yield_devMilk,predict(res))

# tiempo
# metodo              inicio                 fin
# 1   brnn 2026-11-11 15:22:57 2026-11-11 15:23:57
# 2   adam 2026-11-11 15:27:03 2026-11-11 15:29:26
# 
