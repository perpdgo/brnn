\name{un_normalize}
\alias{un_normalize}
\title{un_normalize}
\description{
  Internal function for going back to the original scale.
}
\usage{
   un_normalize(z,base,spread)
}
\arguments{
  \item{z}{a vector or matrix with values normalized between -1 and 1, this vector was obtained when normalizing a vector or matrix x.}
  \item{base}{If z is a vector, base is the minimum of x. If x is a matrix, base is a vector with the minimum for each of the columns of the matrix x.}
  \item{spread}{if z is a vector, spread=base-max(x). If x is a matrix, spread is a vector calculated for each of the columns of x.}
}

\details{
  x=base+0.5*spread*(z+1) 
}
\value{
  A vector or matrix with the resulting un normalized values.
}

\keyword{models}
