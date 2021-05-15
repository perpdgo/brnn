\name{estimate.trace}
\alias{estimate.trace}
\title{estimate.trace}
\description{
  The estimate.trace function estimates the trace of the inverse of a possitive definite and symmetric 
  matrix using the algorithm developed by Bai et al. (1996). It is specially useful 
  when the matrix is huge.
}
\usage{
  estimate.trace(A,tol=1E-6,samples=40,cores=1)
}
\arguments{
  \item{A}{(numeric), positive definite and symmetric matrix.}
  \item{tol}{numeric tolerance, a very small number useful for checking convergenge in the Bai's algorithm.}
  \item{samples}{integer, number of Monte Carlo replicates to estimate the trace of the inverse.}
  \item{cores}{Number of cpu cores to use for calculations (only availible in UNIX-like operating systems).}
}

\references{
Bai, Z. J., M. Fahey and G. Golub. 1996. "Some large-scale matrix computation problems." 
\emph{Journal of Computational and Applied Mathematics}, \bold{74(1-2)}, 71-89.
}

\examples{
\dontrun{
library(brnn)
data(Jersey)

#Estimate the trace of the iverse of G matrix
estimate.trace(G)

#The TRUE value
sum(diag(solve(G)))

}
}
\keyword{matrix}
