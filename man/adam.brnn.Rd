\name{adam.brnn}
\alias{adam.brnn}
\alias{adam.brnn.formula}
\alias{adam.brnn.default}
\alias{cv.adam.brnn}
\alias{get.lambda.seq}
\alias{getopt.lambda}
\alias{initnw_alt}

\title{adam.brnn}
\description{
  The adam.brnn function fits a two layer neural network (one hidden layer with \code{tanh}
  activation and linear output) by minimizing the error sum of squares plus a weight decay
  penalty with the Adam algorithm (Kingma and Ba, 2015). The weight decay parameter can be fixed
  or selected by K-fold cross-validation with \code{cv.adam.brnn}. The functions
  \code{get.lambda.seq}, \code{getopt.lambda} and \code{initnw_alt} are auxiliary functions.
}
\usage{
  adam.brnn(x,\dots)

  \method{adam.brnn}{formula}(formula, data, contrasts=NULL,\dots)

  \method{adam.brnn}{default}(x,y,neurons=2,lambda=NULL,alpha=0.05,beta1=0.9,beta2=0.999,
       eps=1e-8,change=0.001,min_grad=1e-10,epochs=500,normalize=TRUE,
       theta_start=NULL,K=10,start.method="Xavier-Glorot",penalize.bias=FALSE,
       verbose=FALSE,WarmStart=TRUE,lambda.criterion="lambda.min",
       lambda_par=list(lambda_max=10,lambda_min=0.0001,nlambda=10))

  cv.adam.brnn(x,y,neurons,K,WarmStart=TRUE,lambda_seq=NULL,alpha=0.05,beta1=0.9,
       beta2=0.999,eps=1e-8,change=0.001,epochs=500,normalize=TRUE,
       start.method="Xavier-Glorot",penalize.bias=FALSE,
       lambda_par=list(lambda_max=10,lambda_min=0.001,nlambda=10),
       lambda.criterion="lambda.min",verbose=FALSE)

  get.lambda.seq(lambda_max=10,lambda_min=0.001,nlambda=10)

  getopt.lambda(lambda,cvm,cvsd,lambda.criterion="lambda.1se")

  initnw_alt(neurons,p)
}
\arguments{
  \item{formula}{A formula of the form \code{y ~ x1 + x2 + \dots}}
  \item{data}{Data frame from which variables specified in \code{formula} are preferentially to be taken.}
  \item{contrasts}{an optional list of contrasts to be used for some or all of the factors appearing as variables in the model formula.}
  \item{x}{(numeric, \eqn{n \times p}{n x p}) incidence matrix.}
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector (NAs not allowed).}
  \item{neurons}{positive integer that indicates the number of neurons.}
  \item{lambda}{weight decay parameter, \eqn{\lambda\geq 0}{lambda>=0}. If it is a number, the network is fitted using that value.
                If \code{NULL} or a numeric vector, the value is selected by K-fold cross-validation (the vector is used as the grid
                of values). In \code{getopt.lambda} it is the vector of candidate values.}
  \item{alpha}{positive number, learning rate of Adam, default value 0.05.}
  \item{beta1}{exponential decay rate for the first moment estimates, default value 0.9.}
  \item{beta2}{exponential decay rate for the second moment estimates, default value 0.999.}
  \item{eps}{small positive number added to the denominator of the Adam update, default value \eqn{1\times 10^{-8}}{1e-8}.}
  \item{change}{The program will stop if the maximum (in absolute value) of the differences of the F function in 3 consecutive iterations is less than this quantity.}
  \item{min_grad}{minimum gradient.}
  \item{epochs}{positive integer, maximum number of epochs (iterations) to train, default 500.}
  \item{normalize}{logical, if TRUE will normalize inputs and output, the default value is TRUE.}
  \item{theta_start}{optional list of length \code{neurons} with the starting values, the \eqn{k}{k}-th element is the vector
                     \eqn{(w_k, b_k, \beta_1^{[k]},...,\beta_p^{[k]})'}, the same format as the component \code{theta} of the output.
                     If \code{NULL}, the values are generated according to \code{start.method}.}
  \item{K}{integer, number of folds for the cross-validation, default value 10.}
  \item{start.method}{character, method used to initialize the weights and biases, either \code{"Xavier-Glorot"} (default) or \code{"Nguyen-Widrow"}.}
  \item{penalize.bias}{logical, if \code{TRUE} the biases \eqn{b_k}{b_k} of the hidden layer are also penalized, so the penalty is the sum of squares of all
                       the parameters (as in \code{\link{brnn}}). The default value is \code{FALSE}, following the usual practice of weight decay of
                       leaving the biases unregularized, see Details. The values of \code{lambda} are not comparable between both options.}
  \item{verbose}{logical, if TRUE will print iteration history.}
  \item{WarmStart}{logical, if TRUE, in the cross-validation the estimates obtained for a value of \code{lambda} are used as starting values for the next (smaller) one.}
  \item{lambda.criterion}{character, rule used to select \code{lambda}: \code{"lambda.min"} is the value with the smallest cross-validation error and
                          \code{"lambda.1se"} is the largest value with error not greater than the minimum error plus its standard deviation across folds.}
  \item{lambda_par}{list with elements \code{lambda_max}, \code{lambda_min} and \code{nlambda}, used to build the grid of values for \code{lambda} when it is \code{NULL}.}
  \item{lambda_seq}{numeric vector with the grid of values of \code{lambda}, if \code{NULL} it is obtained with \code{get.lambda.seq} and \code{lambda_par}.}
  \item{lambda_max, lambda_min, nlambda}{maximum value, minimum value and number of values of the grid, which is equally spaced on the logarithmic scale and in decreasing order.}
  \item{cvm, cvsd}{numeric vectors with the mean and the standard deviation of the cross-validation error for each value of \code{lambda}.}
  \item{p}{positive integer, number of inputs.}
  \item{\dots}{arguments passed to or from other methods.}
}

\details{
  The model is the same as in \code{\link{brnn}}:

  \eqn{y_i=\sum_{k=1}^s w_k g(b_k + \sum_{j=1}^p x_{ij} \beta_j^{[k]}) + e_i, i=1,...,n}

  \itemize{
    \item{\eqn{e_i \sim N(0,\sigma_e^2)}{e_i ~ N(0,\sigma_e^2)}.}
    \item{\eqn{s} is the number of neurons.}
    \item{\eqn{w_k}{w_k} is the weight of the \eqn{k}{k}-th neuron, \eqn{k=1,...,s}{k=1,...,s}.}
    \item{\eqn{b_k}{b_k} is a bias for the \eqn{k}{k}-th neuron, \eqn{k=1,...,s}{k=1,...,s}.}
    \item{\eqn{\beta_j^{[k]}}{beta_j^{[s]}} is the weight of the \eqn{j}{j}-th input to the net, \eqn{j=1,...,p}{j=1,...,p}.}
    \item{\eqn{g_k(\cdot)}{g_k(.)} is the activation function, in this implementation \eqn{g_k(x)=\frac{\exp(2x)-1}{\exp(2x)+1}}{g_k(x)=(exp(2x)-1)/(exp(2x)+1)}.}
   }

  The software will minimize

  \deqn{F=E_D + \lambda E_W}{F=E_D + lambda*E_W}

  where \eqn{E_D=\sum_{i=1}^n (y_i-\hat y_i)^2}{E_D=sum (y_i-yhat_i)^2} is the error sum of squares and
  \eqn{E_W}{E_W} is the sum of squares of the penalized parameters,

  \deqn{E_W=\sum_{k=1}^s w_k^2+\sum_{k=1}^s\sum_{j=1}^p (\beta_j^{[k]})^2+\delta\sum_{k=1}^s b_k^2}{E_W=sum w_k^2 + sum sum (beta_j^[k])^2 + delta*sum b_k^2}

  with \eqn{\delta=0}{delta=0} if \code{penalize.bias=FALSE} (default) and \eqn{\delta=1}{delta=1} if \code{penalize.bias=TRUE}. By default the
  biases of the hidden layer are not penalized, which follows the usual practice of weight decay in neural networks, where only the weights
  are penalized (Goodfellow et al., 2016, Sec. 7.1; Bishop, 2006, Sec. 5.5.1). With \code{penalize.bias=TRUE} all the weights and biases
  are penalized, as in \code{\link{brnn}}. The gradient of \eqn{F}{F} with respect to \eqn{b_k}{b_k} includes the term
  \eqn{2\delta\lambda b_k}{2*delta*lambda*b_k}. Note that, unlike the intercept of a ridge regression, \eqn{b_k}{b_k} is not a global intercept
  (this model has no overall mean), so the choice is a modeling decision, and the value of \code{lambda} selected by cross-validation depends on it.

  At iteration \eqn{t}{t}, with gradient \eqn{g_t}{g_t} of \eqn{F}{F}, the parameters are updated using Adam:

  \deqn{m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,}{m_t=beta1*m_(t-1)+(1-beta1)*g_t, v_t=beta2*v_(t-1)+(1-beta2)*g_t^2,}
  \deqn{\theta_t=\theta_{t-1}-\alpha\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}.}{theta_t=theta_(t-1)-alpha*(m_t/(1-beta1^t))/(sqrt(v_t/(1-beta2^t))+eps).}

  The computations are done with matrices (all the neurons and observations at once), so optimized libraries such as OpenBLAS can be used.
  The training stops when the gradient is less than \code{min_grad}, the changes of \eqn{F}{F} in 3 consecutive iterations are less
  than \code{change}, \eqn{E_D<0.01}{E_D<0.01} or \code{epochs} iterations are reached.

  When \code{lambda} is \code{NULL} or a vector, \code{cv.adam.brnn} computes, for each value of the grid, the mean squared prediction error
  (in the original scale of \eqn{y}{y}) in each fold; the inputs and output are normalized with the training data of each fold.
  The folds are random, use \code{set.seed} for reproducible results.

  The output has class \code{c("adam.brnn","brnn")}, so methods defined for \code{brnn} objects (for example \code{print} and \code{predict}) can be used.
}
\value{

object of class \code{c("adam.brnn","brnn")} or \code{c("adam.brnn.formula","adam.brnn","brnn")}. It is a list containing:

\item{$theta}{A list containing weights and biases. The first \eqn{s}{s} components of the list contains vectors with the estimated parameters for
              the \eqn{k}{k}-th neuron, i.e. \eqn{(w_k, b_k, \beta_1^{[k]},...,\beta_p^{[k]})'}.}
\item{$lambda}{value of \eqn{\lambda}{lambda} used (given or selected by cross-validation).}
\item{$reason}{String that indicates the stopping criteria for the training process.}
\item{$Ew}{The sum of the squares of the penalized weights (and biases if \code{penalize.bias=TRUE}).}
\item{$penalize.bias}{logical, whether the biases were penalized.}
\item{$Ed}{The sum of the squares between observed and predicted values.}
\item{$F_history}{value of \eqn{F}{F} in each iteration.}
\item{$epoch}{value of the iteration counter when the training stopped.}
\item{$x_base, $x_spread, $y_base, $y_spread}{values used to normalize the data (\code{NULL} if \code{normalize=FALSE}).}

\code{cv.adam.brnn} and \code{getopt.lambda} return the selected value of \eqn{\lambda}{lambda}, \code{get.lambda.seq} a decreasing
numeric vector and \code{initnw_alt} a matrix with \code{neurons} rows and \eqn{p+2}{p+2} columns with the
Xavier-Glorot starting values.
}

\references{

Bishop, C. M. 2006. \emph{Pattern Recognition and Machine Learning}. Springer.

Goodfellow, I., Bengio, Y. and Courville, A. 2016. \emph{Deep Learning}. MIT Press.

Glorot, X. and Bengio, Y. 2010. "Understanding the difficulty of training deep feedforward neural networks",
\emph{Proceedings of the 13th International Conference on Artificial Intelligence and Statistics}, 249-256.

Kingma, D. P. and Ba, J. 2015. "Adam: a method for stochastic optimization",
\emph{Proceedings of the 3rd International Conference on Learning Representations}.

}

\seealso{
	\code{\link{brnn}}, \code{\link{predict.brnn}}
}

\examples{

\dontrun{

library(brnn)

###############################################################
#Example 1 
#Noise triangle wave function, similar to example 1 in Foresee and Hagan (1997)

#Generating the data
x1=seq(0,0.23,length.out=25)
y1=4*x1+rnorm(25,sd=0.1)
x2=seq(0.25,0.75,length.out=50)
y2=2-4*x2+rnorm(50,sd=0.1)
x3=seq(0.77,1,length.out=25)
y3=4*x3-4+rnorm(25,sd=0.1)
x=c(x1,x2,x3)
y=c(y1,y2,y3)

#lambda fixed, formula interface
out=adam.brnn(y~x,neurons=3,lambda=0.001,epochs=1000)

#With the default method the call is (10-fold cross-validation to find lambda,
#2 neurons and 500 epochs)
#out=adam.brnn(y=y,x=x)

plot(x,y)
lines(x,predict(out),col="blue",lty=2)

#lambda selected by 5-fold cross-validation
out=adam.brnn(y=y,x=as.matrix(x),neurons=3,K=5,epochs=1000,
              lambda_par=list(lambda_max=1,lambda_min=1e-5,nlambda=8))

#get optimized lambda
out$lambda

###############################################################
#Example 2
#sin wave function, example in the Matlab 2010b demo.

x = seq(-1,0.5,length.out=100)
y = sin(2*pi*x)+rnorm(length(x),sd=0.1)

#With the formula interface
out=adam.brnn(y~x,neurons=3)

#With the default method the call is
#out=adam.brnn(y=y,x=as.matrix(x),neurons=3)

plot(x,y)
lines(x,predict(out),col="blue",lty=2)

legend("bottomright",legend="Fitted model",col="blue",lty=2,bty="n")

###############################################################
#Example 3
#2 Inputs and 1 output
#the data used in Paciorek and
#Schervish (2004). The data is from a two input one output function with Gaussian noise
#with mean zero and standard deviation 0.25

data(twoinput)

#Formula interface
out=adam.brnn(y~x1+x2,data=twoinput,neurons=10)

#With the default S3 method
#out=adam.brnn(y=as.vector(twoinput$y),x=as.matrix(cbind(twoinput$x1,twoinput$x2)),neurons=10)

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
#Example 4
#Gianola et al. (2011).
#Warning, it will take a while

#Load the Jersey dataset
data(Jersey)

# using y~\Gamma*X model
descomp = eigen(G, symmetric = TRUE)
Gamma = descomp$vectors
Lambda = descomp$values

indice=Lambda>1e-6
Gamma=Gamma[,indice]
Lambda = Lambda[indice]

X = Gamma \%*\% diag(sqrt(Lambda))

#Fit the model with the FULL DATA
#Formula interface
out=adam.brnn(pheno$yield_devMilk~X,neurons=2)

#Obtain predictions and plot them against fitted values
plot(pheno$yield_devMilk,predict(out))


}

}

\keyword{models}
