\name{normalize}
\alias{normalize}
\title{normalize}
\description{
  Internal function for normalizing the data.
  This function makes a linear transformation of the inputs such that the  values lie between -1 and 1.
}
\usage{
   normalize(x,base,spread)
}
\arguments{
  \item{x}{a vector or matrix that needs to be normalized.}
  \item{base}{If x is a vector, base is the minimum of x. If x is a matrix, base is a vector with the minimum for each of the columns of the matrix x.}
  \item{spread}{if x is a vector, spread=base-max(x). If x is a matrix, spread is a vector calculated for each of the columns of x.}
}

\details{
  z=2*(x-base)/spread - 1 
}
\value{
  A vector or matrix with the resulting normalized values.
}

\keyword{models}
