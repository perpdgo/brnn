\name{initnw}
\alias{initnw}
\title{Initialize networks weights and biases}
\description{
  Function to initialize the weights and biases in a neural network. It uses the Nguyen-Widrow (1990) algorithm.
}
\usage{
     initnw(neurons,p,n,npar)
}
\arguments{
  \item{neurons}{Number of neurons.}
  \item{p}{Number of predictors.}
  \item{n}{Number of cases.}
  \item{npar}{Number of parameters to be estimate including only weights and biases, and should be equal to \eqn{neurons \times (1+1+p)+1}{neurons*(1+1+p)+1}.}
}

\details{

  The algorithm is described in Nguyen-Widrow (1990) and in other books, see for example Sivanandam and Sumathi (2005). The algorithm is briefly described below.
 
  \itemize{
    \item{1.-Compute the scaling factor \eqn{\theta=0.7 p^{1/n}}{theta=0.7*p^(1/n)}.}
    \item{2.- Initialize the weight and biases for each neuron at random, for example generating random numbers from \eqn{U(-0.5,0.5)}{U(-0.5,0.5)}.}
    \item{3.- For each neuron:}
      \itemize{
         \item{compute \eqn{\eta_k=\sqrt{\sum_{j=1}^p (\beta_j^{(k)})^2}},}
         \item{update \eqn{(\beta_1^{(k)},...,\beta_p^{(k)})'},}
          \deqn{\beta_j^{(k)}=\frac{\theta \beta_j^{(k)}}{\eta_k}, j=1,...,p,}
         \item{Update the bias \eqn{(b_k)}{(b_k)} generating a random number from \eqn{U(-\theta,\theta)}{U(-theta,theta)}.}
      }
  }
}

\value{
  A list containing initial values for weights and biases. The first \eqn{s}{s} components of the list contains vectors with the initial values for 
              the weights and biases of the \eqn{k}{k}-th neuron, i.e. \eqn{(\omega_k, b_k, \beta_1^{(k)},...,\beta_p^{(k)})'}.
}

\references{
Nguyen, D. and Widrow, B. 1990. "Improving the learning speed of 2-layer neural networks by choosing initial values of the adaptive weights",
\emph{Proceedings of the IJCNN}, \bold{3}, 21-26.

Sivanandam, S.N. and Sumathi, S. 2005. Introduction to Neural Networks Using MATLAB 6.0. Ed. McGraw Hill, First edition. 

}


\examples{
\dontrun{
#Load the library
library(brnn)

#Set parameters
neurons=3
p=4
n=10
npar=neurons*(1+1+p)+1
initnw(neurons=neurons,p=p,n=n,npar=npar)

}
}

\keyword{models}

