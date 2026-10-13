\name{predict.brnn_nominal}
\alias{predict.brnn_nominal}

\title{Predictions from a Nominal Bayesian Regularized Neural Network}

\description{
Produces class or probability predictions from an object of class
\code{"brnn_nominal"}.
}

\usage{
\method{predict}{brnn_nominal}(
  object,
  newdata = NULL,
  type = c("class", "prob"),
  \dots
)
}

\arguments{
  \item{object}{
    An object of class \code{"brnn_nominal"} returned by
    \code{brnn_nominal()}.
  }

  \item{newdata}{
    Optional numeric matrix or data frame containing new predictor
    observations. If \code{NULL}, fitted predictions from the training
    data are returned.
  }

  \item{type}{
    Character string indicating the type of prediction. Use
    \code{"class"} for predicted classes or \code{"prob"} for class
    probabilities.
  }


  \item{\dots}{
    Additional arguments. Currently not used.
  }
}

\details{
This function is the \code{predict()} method for objects of class
\code{"brnn_nominal"}.

When \code{type = "class"}, the class with the highest estimated
probability is returned. When \code{type = "prob"}, the estimated
probabilities for all response classes are returned.

If \code{newdata = NULL}, predictions for the training observations are
returned. For models fitted through the default interface, the number and
order of predictor variables in \code{newdata} must agree with those used
to fit the model. For formula models, the stored terms, contrasts, and
factor levels are used to construct the predictor matrix.
}

\value{
If \code{type = "class"}, a factor containing the predicted classes.

If \code{type = "prob"}, a numeric matrix in which rows correspond to
observations and columns correspond to response classes.
}

\seealso{
  \code{\link{brnn_nominal}}
}

\examples{
x <- as.matrix(iris[, 1:4])
y <- iris$Species

fit <- brnn_nominal(
  x = x,
  y = y,
  neurons = 2,
  epochs = 10,
  verbose = FALSE,
  seed = 123
)

predicted_class <- predict(fit, type = "class")
predicted_prob <- predict(fit, type = "prob")

head(predicted_class)
head(predicted_prob)
}

\keyword{models}
\keyword{classification}
