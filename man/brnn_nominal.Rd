\name{brnn_nominal}
\alias{brnn_nominal}
\alias{brnn_nominal.default}
\alias{brnn_nominal.formula}
\alias{print.brnn_nominal}
\alias{summary.brnn_nominal}
\alias{coef.brnn_nominal}
\title{Bayesian Regularized Neural Network for Nominal Classification}

\description{
Fits a two-layer feed-forward neural network for nominal classification.
The hidden layer uses a user-selected activation function, while the output
layer uses the softmax function. Model parameters are estimated using
cross-entropy loss, adaptive Bayesian regularization, and a
Levenberg--Marquardt optimization scheme.
}

\usage{
brnn_nominal(x, \dots)

\method{brnn_nominal}{formula}(
  formula,
  data,
  contrasts = NULL,
  \dots
)

\method{brnn_nominal}{default}(
  x,
  y,
  neurons = 3,
  activation = "tanh",
  normalize = TRUE,
  epochs = 200,
  alpha_init = 0.01,
  mu = 0.005,
  mu_dec = 0.1,
  mu_inc = 10,
  mu_max = 1e10,
  min_grad = 1e-6,
  change = 1e-6,
  verbose = TRUE,
  seed = 123,
  \dots
)

\method{summary}{brnn_nominal}(object, \dots)

\method{coef}{brnn_nominal}(object, \dots)
}

\arguments{

  \item{formula}{
   	A formula of the form \code{y ~ x1 + x2 + \dots}, where the
  	response represents the nominal classes.
  }

  \item{data}{
  	A data frame containing the variables specified in \code{formula}.
  }

  \item{contrasts}{
  	An optional list of contrasts to be used for factor predictors in
  	the model formula.
  }
  
  \item{x}{
    A numeric matrix or data frame containing the predictor variables.
    Rows correspond to observations and columns to predictors.
  }

  \item{y}{
    A factor or vector containing the nominal response classes.
  }

  \item{neurons}{
    Positive integer indicating the number of neurons in the hidden layer.
  }

  \item{activation}{
    Character string indicating the hidden-layer activation function.
    Available options are \code{"tanh"}, \code{"sigmoid"}, and
    \code{"relu"}.
  }

  \item{normalize}{
    Logical. If \code{TRUE}, predictor variables are normalized before
    fitting the model.
  }

  \item{epochs}{
    Positive integer indicating the maximum number of training epochs.
  }

  \item{alpha_init}{
    Positive numeric value specifying the initial Bayesian regularization
    parameter.
  }

  \item{mu}{
    Positive numeric value specifying the initial Levenberg--Marquardt
    damping parameter.
  }

  \item{mu_dec}{
    Positive numeric value used to decrease the damping parameter after
    an accepted optimization step.
  }

  \item{mu_inc}{
    Positive numeric value used to increase the damping parameter after
    a rejected optimization step.
  }

  \item{mu_max}{
    Maximum value allowed for the damping parameter before training is
    stopped.
  }

  \item{min_grad}{
    Minimum gradient norm used as a stopping criterion.
  }

  \item{change}{
    Minimum absolute change in the regularized objective function used as
    a stopping criterion.
  }

  \item{verbose}{
    Logical. If \code{TRUE}, information about the training process is
    printed.
  }

  \item{seed}{
    Integer used to initialize the random number generator before the
    network parameters are generated.
  }

  \item{object}{
    An object of class \code{"brnn_nominal"} returned by
    \code{brnn_nominal()}.
  }


  \item{\dots}{
    Additional arguments passed to or from methods.
  }
}

\details{
  The software fits a two-layer Bayesian regularized neural network for
  nominal response data. For observation \eqn{i}{i} and class
  \eqn{\ell}{l}, the estimated probability is
  
  \deqn{
  \pi_{\ell}(\boldsymbol{x}_i;\boldsymbol{\theta})
  =
  \frac{\exp(o_i^{(\ell)})}
  {\sum_{t=1}^{K}\exp(o_i^{(t)})},
  \qquad
  o_i^{(\ell)}
  =
  w_0^{(\ell)}
  +
  \sum_{k=1}^{s}
  w_k^{(\ell)}
  g\left(
  \beta_0^{(k)}
  +
  \sum_{j=1}^{p}x_{ij}\beta_j^{(k)}
  \right)
  }{
  pi_l(x_i;theta) =
  exp(o_i^(l)) / sum_t exp(o_i^(t))
  }
  
  where:
  
  \itemize{
    \item \eqn{K}{K} is the number of nominal classes.
    \item \eqn{s}{s} is the number of hidden neurons.
    \item \eqn{\beta_0^{(k)}}{beta0^(k)} and
          \eqn{\beta_j^{(k)}}{beta_j^(k)} are the bias and input weights
          of hidden neuron \eqn{k}{k}.
    \item \eqn{w_0^{(\ell)}}{w0^(l)} and
          \eqn{w_k^{(\ell)}}{w_k^(l)} are the bias and weights associated
          with class \eqn{\ell}{l}.
    \item \eqn{g(\cdot)}{g(.)} is the hidden-layer activation function.
    \item \eqn{\boldsymbol{\theta}}{theta} contains all network weights
          and biases.
  }
  
  The nominal response is represented through one-hot encoding, and the
  softmax function ensures that the estimated class probabilities are
  positive and sum to one.
  
  The software minimizes
  
  \deqn{
  F(\boldsymbol{\theta})
  =
  \mathcal{L}(\boldsymbol{\theta})
  +
  \alpha E_W(\boldsymbol{\theta})
  }{
  F(theta) = L(theta) + alpha E_W(theta)
  }
  
  where:
  
  \itemize{
    \item \eqn{
      \mathcal{L}(\boldsymbol{\theta})
      =
      -\sum_{i=1}^{n}\sum_{\ell=1}^{K}
      y_{i\ell}\log
      \pi_{\ell}(\boldsymbol{x}_i;\boldsymbol{\theta})
    }{
      L(theta) = -sum_i sum_l y_il log pi_l(x_i;theta)
    } is the multiclass cross-entropy loss.

    \item \eqn{
      E_W(\boldsymbol{\theta})
      =
      \frac{1}{2}\sum_{r=1}^{q}\theta_r^2
    }{
      E_W(theta) = 0.5 sum_r theta_r^2
    } is the quadratic penalty on weights and biases.

    \item \eqn{\alpha}{alpha} is the Bayesian regularization parameter,
          which is updated adaptively using the effective number of
          parameters \eqn{\gamma}{gamma}.
  }
  
  Model parameters are estimated using a Levenberg--Marquardt scheme with
  a Gauss--Newton/Fisher approximation to the curvature matrix.
}

\value{

The default interface returns an object of class
\code{"brnn_nominal"}. The formula interface returns an object with
classes \code{"brnn_nominal.formula"} and \code{"brnn_nominal"}.

The fitted object is represented by a list containing the following main
components:

\item{\code{theta}}{
  Estimated weights and biases of the neural network.
}

\item{\code{alpha}}{
  Final value of the Bayesian regularization parameter.
}

\item{\code{gamma}}{
  Estimated effective number of parameters.
}

\item{\code{epochs_used}}{
  Number of training epochs performed.
}

\item{\code{reason}}{
  Character string describing the stopping criterion.
}

\item{\code{converged}}{
  Logical value indicating whether a convergence criterion was satisfied.
}

\item{\code{activation}}{
  Hidden-layer activation function.
}

\item{\code{neurons}}{
  Number of hidden neurons.
}

\item{\code{classes}}{
  Levels of the nominal response.
}

\item{\code{fitted_prob}}{
  Matrix of fitted class probabilities.
}

\item{\code{fitted_class}}{
  Factor containing the fitted classes.
}

\item{\code{F_history}, \code{L_history}, \code{Ew_history}}{
  Histories of the regularized objective function, cross-entropy loss,
  and weight penalty.
}

\item{\code{alpha_history}, \code{gamma_history}}{
  Histories of the regularization parameter and the effective number of
  parameters.
}

\item{\code{mu_history}, \code{grad_history}}{
  Histories of the damping parameter and gradient norm.
}

\item{\code{x_base}, \code{x_spread}}{
  Values used to normalize new predictor observations.
}
For models fitted through the formula interface, the object also contains
\code{terms}, \code{coefnames}, \code{call}, \code{contrasts}, and
\code{xlevels}.

The method \code{coef()} returns a named numeric vector containing all
estimated network parameters. The method \code{print()} returns the fitted
object invisibly after displaying a brief description.
}

\references{

Bishop, C. M. (2006).
\emph{Pattern Recognition and Machine Learning}.
Springer.

Foresee, F. D. and Hagan, M. T. (1997).
Gauss-Newton approximation to Bayesian learning.
\emph{Proceedings of the International Conference on Neural Networks},
volume 3, 1930--1935.

Hagan, M. T. and Menhaj, M. B. (1994).
Training feedforward networks with the Marquardt algorithm.
\emph{IEEE Transactions on Neural Networks}, \bold{5}(6), 989--993.

Levenberg, K. (1944).
A method for the solution of certain non-linear problems in least squares.
\emph{Quarterly of Applied Mathematics}, \bold{2}(2), 164--168.

MacKay, D. J. C. (1992).
Bayesian interpolation.
\emph{Neural Computation}, \bold{4}(3), 415--447.

Marquardt, D. W. (1963).
An algorithm for least-squares estimation of nonlinear parameters.
\emph{Journal of the Society for Industrial and Applied Mathematics},
\bold{11}(2), 431--441.

}
\seealso{
  \code{\link{predict.brnn_nominal}},
  \code{\link{DryBean}}
}

\examples{

\dontrun{

library(brnn)

########################################
#Example 1, classic iris dataset 
########################################

x <- as.matrix(iris[, 1:4])
y <- iris$Species

# Default interface
fit_default <- brnn_nominal(
  x = x,
  y = y,
  neurons = 2,
  epochs = 1000,
  verbose =TRUE,
  seed = 123
)

print(fit_default)
summary(fit_default)
head(predict(fit_default, type = "class"))
head(coef(fit_default))

# Formula interface
fit_formula <- brnn_nominal(
  Species ~ .,
  data = iris,
  neurons = 2,
  epochs = 1000,
  verbose = TRUE,
  seed = 123
)

head(
  predict(
    fit_formula,
    newdata = iris[1:5, ],
    type = "prob"
  )
)

########################################
#Example 2, DryBean dataset
########################################

data(DryBean)

x<- as.matrix(DryBean[,1:16])
y<- DryBean$Class

# Default interface
fit_default <- brnn_nominal(
  x = x,
  y = y,
  neurons = 2,
  epochs = 1000,
  verbose = TRUE,
  seed = 123,
)

fit_default

# Formula interface
fit_formula <- brnn_nominal(
  Class ~ .,
  data = DryBean,
  neurons = 2,
  epochs = 1000,
  verbose = TRUE,
  seed = 123
)

fit_formula

#Training and testing, assign 80% of observations to training and 20% testing
trn_tst<-rep("trn",nrow(DryBean))
index<-sample(1:nrow(DryBean),size=as.integer(0.20*nrow(DryBean)))
trn_tst[index]<-"tst"

fit_default <- brnn_nominal(
  x = x[trn_tst=="trn",],
  y = y[trn_tst=="trn"],
  neurons = 2,
  epochs = 1000,
  verbose = TRUE,
  seed = 123,
)

predictions<-predict(fit_default,newdata=x[trn_tst=="tst",],type="class")

table(predictions,y[trn_tst=="tst"])

#
fit_formula <- brnn_nominal(
  Class ~ .,
  data = subset(DryBean,trn_tst=="trn"),
  neurons = 2,
  epochs = 1000,
  verbose = TRUE,
  seed = 123
)

#predict(fit_default,newdata=subset(DryBean,trn_tst=="tst"),type="class")
}

}

\keyword{models}
\keyword{classification}
