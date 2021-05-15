\name{brnn_extended}
\alias{brnn_extended}
\alias{brnn_extended.formula}
\alias{brnn_extended.default}
\alias{coef.brnn_extended}
\alias{print.brnn_extended}
\alias{summary.brnn_extended}

\title{brnn_extended}
\description{
  The brnn_extended function
  fits a two layer neural network as described in MacKay (1992) and Foresee and Hagan (1997). It uses the 
  Nguyen and Widrow algorithm (1990) to assign initial weights and the Gauss-Newton algorithm to 
  perform the optimization. The hidden layer contains two groups of neurons 
  that allow us to assign different prior distributions for two groups of input variables.
  
}
\usage{
  brnn_extended(x, \dots)

  \method{brnn_extended}{formula}(formula, data, contrastsx=NULL,contrastsz=NULL,\dots)

  \method{brnn_extended}{default}(x,y,z,neurons1,neurons2,normalize=TRUE,epochs=1000,
              mu=0.005,mu_dec=0.1, mu_inc=10,mu_max=1e10,min_grad=1e-10,
              change = 0.001, cores=1,verbose =FALSE,\dots)
}
\arguments{
  \item{formula}{A formula of the form \code{y ~ x1 + x2 \dots | z1 + z2 \dots}, the | is used to separate the two groups of input variables.}
  \item{data}{Data frame from which variables specified in  \code{formula} are preferentially to be taken.}
  \item{y}{(numeric, \eqn{n}{n}) the response data-vector (NAs not  allowed).}
  \item{x}{(numeric, \eqn{n \times p}{n x p}) incidence matrix for variables in group 1.}
  \item{z}{(numeric, \eqn{n \times q}{n x q}) incidence matrix for variables in group 2.}
  \item{neurons1}{positive integer that indicates the number of neurons for variables in group 1.}
  \item{neurons2}{positive integer that indicates the number of neurons for variables in group 2.}
  \item{normalize}{logical, if TRUE will normalize inputs and output, the default value is TRUE.}
  \item{epochs}{positive integer, maximum number of epochs to train, default 1000.}
  \item{mu}{positive number that controls the behaviour of the Gauss-Newton optimization algorithm, default value 0.005.}
  \item{mu_dec}{positive number, is the mu decrease ratio, default value 0.1.}
  \item{mu_inc}{positive number, is the mu increase ratio, default value 10.}
  \item{mu_max}{maximum mu before training is stopped, strict positive number, default value \eqn{1\times 10^{10}}{1e10}.}
  \item{min_grad}{minimum gradient.}
  \item{change}{The program  will stop if the maximum (in absolute value) of the differences of the F 
                function in 3 consecutive iterations is less than this quantity.}
  \item{cores}{Number of cpu cores to use for calculations (only available in UNIX-like operating systems). The function detectCores in the R package 
               parallel can be used to attempt to detect the number of CPUs in the machine that R is running, but not necessarily 
               all the cores are available for the current user, because for example in multi-user 
               systems it will depend on system policies. Further details can be found in the documentation for the parallel package}
  \item{verbose}{logical, if TRUE will print iteration history.}
  \item{contrastsx}{an optional list of contrasts to be used for some or 
                    all of the factors appearing as variables in the first group of input variables in the model formula.}
  \item{contrastsz}{an optional list of contrasts to be used for some or 
                    all of the factors appearing as variables in the second group of input variables in the model formula.}
  
  \item{\dots}{arguments passed to or from other methods.}
}

\details{
  The software fits a two layer network as described in MacKay (1992) and Foresee and Hagan (1997). 
  The model is given by:
  
  \eqn{y_i= \sum_{k=1}^{s_1} w_k^{1} g_k (b_k^{1} + \sum_{j=1}^p x_{ij} \beta_j^{1[k]}) +
            \sum_{k=1}^{s_2} w_k^{2} g_k (b_k^{2} + \sum_{j=1}^q z_{ij} \beta_j^{2[k]})\,\,e_i, i=1,...,n}
  
  \itemize{
    \item{\eqn{e_i \sim N(0,\sigma_e^2)}{e_i ~ N(0,\sigma_e^2)}.}
    \item{\eqn{g_k(\cdot)}{g_k(.)} is the activation function, in this implementation \eqn{g_k(x)=\frac{\exp(2x)-1}{\exp(2x)+1}}{g_k(x)=(exp(2x)-1)/(exp(2x)+1)}.}
  }
  
  The software will minimize 
 
  \deqn{F=\beta E_D + \alpha \theta_1' \theta_1 +\delta \theta_2' \theta_2 }
  
  where 
  \itemize{
     \item{\eqn{E_D=\sum_{i=1}^n (y_i-\hat y_i)^2}{y_i-\hat y_i}, i.e. the sum of squared errors.}
     \item{\eqn{\beta=\frac{1}{2\sigma^2_e}}{beta=1/(2*sigma^2_e)}.}
     \item{\eqn{\alpha=\frac{1}{2\sigma_{\theta_1}^2}}{alpha=1/(2*sigma_theta_1^2)}, \eqn{\sigma_{\theta_1}^2}{sigma_theta_1^2} is a dispersion parameter for weights and biases for the associated to 
          the first group of neurons.}
     \item{\eqn{\delta=\frac{1}{2\sigma_{\theta_2}^2}}{delta=1/(2*sigma_theta_2^2)}, \eqn{\sigma_{\theta_2}^2}{sigma_theta_2^2} is a dispersion parameter for weights and biases for the associated to
          the second group of neurons.}
   }
}
\value{

object of class \code{"brnn_extended"} or \code{"brnn_extended.formula"}. Mostly internal structure, but it is a list containing:

\item{$theta1}{A list containing weights and biases. The first \eqn{s_1}{s_1} components of the list contain vectors with 
               the estimated parameters for the \eqn{k}{k}-th neuron, i.e. \eqn{(w_k^1, b_k^1, \beta_1^{1[k]},...,\beta_p^{1[k]})'}. 
               \eqn{s_1}{s_1} corresponds to neurons1 in the argument list.}
\item{$theta2}{A list containing weights and biases. The first \eqn{s_2}{s_2} components of the list contains vectors with 
               the estimated parameters for the \eqn{k}{k}-th neuron, i.e. \eqn{(w_k^2, b_k^2, \beta_1^{2[k]},...,\beta_q^{2[k]})'}.
                \eqn{s_2}{s_2} corresponds to neurons2 in the argument list.}
\item{$message}{String that indicates the stopping criteria for the training process.}
}

\references{

Foresee, F. D., and M. T. Hagan. 1997. "Gauss-Newton approximation to Bayesian regularization", 
\emph{Proceedings of the 1997 International Joint Conference on Neural Networks}.

MacKay, D. J. C. 1992. "Bayesian interpolation", \emph{Neural Computation}, \bold{4(3)}, 415-447.

Nguyen, D. and Widrow, B. 1990. "Improving the learning speed of 2-layer neural networks by choosing initial values of the adaptive weights",
\emph{Proceedings of the IJCNN}, \bold{3}, 21-26.

}

\seealso{
        \code{\link{predict.brnn_extended}}
}

\examples{

\dontrun{

#Example 5
#Warning, it will take a while

#Load the Jersey dataset
data(Jersey)

#Predictive power of the model using the SECOND set for 10 fold CROSS-VALIDATION
data=pheno
data$G=G
data$D=D
data$partitions=partitions

#Fit the model for the TESTING DATA for Additive + Dominant
out=brnn_extended(yield_devMilk ~ G | D,
                                  data=subset(data,partitions!=2),
                                  neurons1=2,neurons2=2,epochs=100,verbose=TRUE)

#Plot the results
#Predicted vs observed values for the training set
par(mfrow=c(2,1))
yhat_R_training=predict(out)
plot(out$y,yhat_R_training,xlab=expression(hat(y)),ylab="y")
cor(out$y,yhat_R_training)

#Predicted vs observed values for the testing set
newdata=subset(data,partitions==2,select=c(D,G))
ytesting=pheno$yield_devMilk[partitions==2]
yhat_R_testing=predict(out,newdata=newdata)
plot(ytesting,yhat_R_testing,xlab=expression(hat(y)),ylab="y")
cor(ytesting,yhat_R_testing)
  
}
 
}

\keyword{models}
