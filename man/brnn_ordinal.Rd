\name{brnn_ordinal}
\alias{brnn_ordinal}
\alias{brnn_ordinal.formula}
\alias{brnn_ordinal.default}
\alias{print.brnn_ordinal}
\alias{summary.brnn_ordinal}

\title{brnn_ordinal}
\description{
  The brnn_ordinal function fits a Bayesian Regularized Neural Network for Ordinal data.
}
\usage{

  brnn_ordinal(x, \dots)
  
  \method{brnn_ordinal}{formula}(formula, data, contrasts=NULL,\dots)
  
  \method{brnn_ordinal}{default}(x,
               y,
               neurons=2,
               normalize=TRUE,
               epochs=1000,
               mu=0.005,
               mu_dec=0.1,
               mu_inc=10,
               mu_max=1e10,
               min_grad=1e-10,
               change_F=0.01,
               change_par=0.01,
               iter_EM=1000,
               verbose=FALSE,
               \dots)
}

\arguments{
  \item{formula}{A formula of the form \code{y ~ x1 + x2 + \dots}}
  \item{data}{Data frame from which variables specified in  \code{formula} are preferentially to be taken.}
  \item{x}{(numeric, \eqn{n \times p}{n x p}) incidence matrix.}
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector (NAs not  allowed).}
  \item{neurons}{positive integer that indicates the number of neurons.}
  \item{normalize}{logical, if TRUE will normalize inputs and output, the default value is TRUE.}
  \item{epochs}{positive integer, maximum number of epochs(iterations) to train, default 1000.}
  \item{mu}{positive number that controls the behaviour of the Gauss-Newton optimization algorithm, default value 0.005.}
  \item{mu_dec}{positive number, is the mu decrease ratio, default value 0.1.}
  \item{mu_inc}{positive number, is the mu increase ratio, default value 10.}
  \item{mu_max}{maximum mu before training is stopped, strict positive number, default value \eqn{1\times 10^{10}}{1e10}.}
  \item{min_grad}{minimum gradient.}
  \item{change_F}{the program  will stop if the maximum (in absolute value) of the differences of the F function in 3 consecutive iterations is less than this quantity.}
  \item{change_par}{the program will stop iterations of the EM algorithm when the maximum of absolute values of differences between parameters in two consecutive iterations ins less than this quantity.}
  \item{iter_EM}{positive integer, maximum number of iteration for the EM algorithm.}
  \item{verbose}{logical, if TRUE will print iteration history.}
  \item{contrasts}{an optional list of contrasts to be used for some or all of the factors appearing as variables in the model formula.}
  \item{\dots}{arguments passed to or from other methods.}
}

\details{
    The software fits a Bayesian Regularized Neural Network for Ordinal data. The model is an 
    extension of the two layer network as described in MacKay (1992); Foresee and Hagan (1997), and
    Gianola et al. (2011). We use the latent variable approach described in Albert and Chib (1993)
    to model ordinal data, the Expectation maximization (EM) and Levenberg-Marquardt 
    algorithm (Levenberg, 1944; Marquardt, 1963) to fit the model. 
    
    Following Albert and Chib (1993), suppose that \eqn{Y_1,...,Y_n}{Y_1,...,Y_n} are
    observed and \eqn{Y_i}{Y_i} can take values on \eqn{L}{L} ordered values. We are interested
    in modelling the probability \eqn{p_{ij}=P(Y_i=j)}{p_ij=P(Y_i=j)} using the covariates 
    \eqn{x_{i1},...,x_{ip}}{x_i1,...,x_ip}. Let 
    
    \eqn{g(\boldsymbol{x}_i)= \sum_{k=1}^s w_k g_k (b_k + \sum_{j=1}^p x_{ij} \beta_j^{[k]})}, 
    
    where: 
    
    \itemize{
    \item{\eqn{s} is the number of neurons.}
    \item{\eqn{w_k}{w_k} is the weight of the \eqn{k}{k}-th neuron, \eqn{k=1,...,s}{k=1,...,s}.}
    \item{\eqn{b_k}{b_k} is a bias for the \eqn{k}{k}-th neuron, \eqn{k=1,...,s}{k=1,...,s}.}
    \item{\eqn{\beta_j^{[k]}}{beta_j^{[s]}} is the weight of the \eqn{j}{j}-th input to the net, \eqn{j=1,...,p}{j=1,...,p}.}
    \item{\eqn{g_k(\cdot)}{g_k(.)} is the activation function, in this implementation \eqn{g_k(x)=\frac{\exp(2x)-1}{\exp(2x)+1}}{g_k(x)=(exp(2x)-1)/(exp(2x)+1)}.}
   }
   
   Let 

\eqn{Z_i=g(\boldsymbol{x}_i)+e_i},

where:  

\itemize{
  \item \eqn{e_i \sim N(0,1)}{e_i ~ N(0,1)}.
  \item \eqn{Z_i}{Z_i} is an unobserved (latent variable).
}

The output from the model for latent variable is related to 
observed data using the approach employed in the probit 
and logit ordered models, that is \eqn{Y_i=j} if 
\eqn{\lambda_{j-1}<Z_i<\lambda_{j}}, where \eqn{\lambda_j}
are a set of unknown thresholds. We assign prior distributions 
to all unknown quantities (see Albert and Chib, 1993; Gianola et al., 2011) 
for further details. The Expectation maximization (EM) and Levenberg-Marquardt 
algorithm (Levenberg, 1944; Marquardt, 1963) to fit the model.

}

\value{
object of class \code{"brnn_ordinal"}. Mostly internal structure, but it is a list containing:

\item{$theta}{A list containing weights and biases. The first \eqn{s}{s} components of the list contains vectors with the estimated parameters for
              the \eqn{k}{k}-th neuron, i.e. \eqn{(w_k, b_k, \beta_1^{[k]},...,\beta_p^{[k]})'}.}
\item{$threshold}{A vector with estimates of thresholds.}
\item{$alpha}{\eqn{\alpha}{alpha} parameter.}
\item{$gamma}{effective number of parameters.}

}

\references{

Albert J, and S. Chib. 1993. Bayesian Analysis of Binary and Polychotomus Response Data. \emph{JASA}, \bold{88}, 669-679.

Foresee, F. D., and M. T. Hagan. 1997. "Gauss-Newton approximation to Bayesian regularization", 
\emph{Proceedings of the 1997 International Joint Conference on Neural Networks}.

Gianola, D. Okut, H., Weigel, K. and Rosa, G. 2011. "Predicting complex quantitative traits with Bayesian neural networks: a case study with Jersey cows and wheat". \emph{BMC Genetics}, 
\bold{12},87.

Levenberg, K. 1944. "A method for the solution of certain problems in least squares", \emph{Quart. Applied Math.}, \bold{2}, 164-168. 

MacKay, D. J. C. 1992. "Bayesian interpolation", \emph{Neural Computation}, \bold{4(3)}, 415-447.

Marquardt, D. W. 1963. "An algorithm for least-squares estimation of non-linear parameters". \emph{SIAM Journal on Applied Mathematics}, \bold{11(2)}, 431-441. 

}

\seealso{
	\code{\link{predict.brnn_ordinal}}
}

\examples{

\dontrun{
#Load the library
library(brnn)

#Load the dataset
data(GLS)

#Subset of data for location Harare
HarareOrd=subset(phenoOrd,Loc=="Harare")

#Eigen value decomposition for GOrdm keep those 
#eigen vectors whose corresponding eigen-vectors are bigger than 1e-10
#and then compute principal components

evd=eigen(GOrd)
evd$vectors=evd$vectors[,evd$value>1e-10]
evd$values=evd$values[evd$values>1e-10]
PC=evd$vectors%*%sqrt(diag(evd$values))
rownames(PC)=rownames(GOrd)

#Response variable
y=phenoOrd$rating
gid=as.character(phenoOrd$Stock)

Z=model.matrix(~gid-1)
colnames(Z)=gsub("gid","",colnames(Z))

if(any(colnames(Z)!=rownames(PC))) stop("Ordering problem\n")

#Matrix of predictors for Neural net
X=Z\%*\%PC

#Cross-validation
set.seed(1)
testing=sample(1:length(y),size=as.integer(0.10*length(y)),replace=FALSE)
isNa=(1:length(y)\%in\%testing)
yTrain=y[!isNa]
XTrain=X[!isNa,]
nTest=sum(isNa)

neurons=2
	
fmOrd=brnn_ordinal(XTrain,yTrain,neurons=neurons,verbose=FALSE)

#Predictions for testing set
XTest=X[isNa,]
predictions=predict(fmOrd,XTest)
predictions


}
}

\keyword{models}

