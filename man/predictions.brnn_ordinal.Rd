\name{predict.brnn_ordinal}
\alias{predict.brnn_ordinal}
\title{predict.brnn_ordinal}
\description{
  The function produces the predictions for a two-layer feed-forward neural network for ordinal data.
}
\usage{
   \method{predict}{brnn_ordinal}(object,newdata,\dots)
}
\arguments{
  \item{object}{an object of the class \code{brnn_ordinal} as returned by \code{brnn_ordinal}}
  \item{newdata}{matrix or data frame of test examples. A vector is considered to be
                 a row vector comprising a single case.}
  \item{\dots}{arguments passed to or from other methods.}
 }

\details{
  This function is a method for the generic function
  \code{predict()} for class \code{"brnn_ordinal"}.
   It can be invoked by calling \code{predict(x)} for an
   object \code{x} of the appropriate class, or directly by
   calling \code{predict.brnn_ordinal(x)} regardless of the class of the object.
}

\value{

A list with components:

  \item{class}{Predicted class (an integer).}
  \item{probability}{Posterior probability of belonging to a class given the covariates.}
}

\keyword{models}
