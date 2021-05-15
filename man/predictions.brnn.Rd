\name{predict.brnn}
\alias{predict.brnn}
\title{predict.brnn}
\description{
  The function produces the predictions for a two-layer feed-forward neural network.
}
\usage{
   \method{predict}{brnn}(object,newdata,\dots)
}
\arguments{
  \item{object}{an object of the class \code{brnn} as returned by \code{brnn}}
  \item{newdata}{matrix or data frame of test examples. A vector is considered to be
                 a row vector comprising a single case.}
  \item{\dots}{arguments passed to or from other methods.}
 }

\details{
  This function is a method for the generic function
  \code{predict()} for class \code{"brnn"}.
	It can be invoked by calling \code{predict(x)} for an
	object \code{x} of the appropriate class, or directly by
	calling \code{predict.brnn(x)} regardless of the class of the object.
}

\value{
  A vector containing the predictions
}

\keyword{models}
