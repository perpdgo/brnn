# file brnn/brnn_nominal.R
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 or 3 of the License
#  (at your option).
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  A copy of the GNU General Public License is available at
#  http://www.r-project.org/Licenses/

#A package for Bayesian Regularized Neural Networks
#Author: Elmer Nava Sanchez

#Texcoco, Mexico, Sep. 2026

#WARNING: This is an experimental version

# Normalization to the [-1, 1] interval
normalize_nominal <- function(x, base = NULL, spread = NULL) {
  if (is.null(base)) {
    if (is.matrix(x)) {
      base <- apply(x, 2, min)
    } else {
      base <- min(x)
    }
  }
  
  if (is.null(spread)) {
    if (is.matrix(x)) {
      spread <- apply(x, 2, max) - base
    } else {
      spread <- max(x) - base
    }
  }
  
  spread[spread == 0] <- 1
  
  if (is.matrix(x)) {
    return(
      sweep(
        sweep(x, 2, base, "-"),
        2,
        2 / spread,
        "*"
      ) - 1
    )
  }
  
  2 * (x - base) / spread - 1
}


# Sigmoid activation function
sigmoid_nominal <- function(x) {
  1 / (1 + exp(-x))
}


# one-hot encoding for the response variable
one_hot_nominal <- function(y) {
  y <- as.factor(y)
  classes <- levels(y)
  K <- length(classes)
  n <- length(y)
  
  Y <- matrix(
    0,
    nrow = n,
    ncol = K
  )
  
  colnames(Y) <- classes
  
  for (k in seq_len(K)) {
    Y[, k] <- as.integer(y == classes[k])
  }
  
  list(
    Y = Y,
    classes = classes
  )
}


# Derivative of the activation function for the hidden layer
activation_prime_matrix_nominal <- function(
    U,
    activation = "tanh"
) {
  activation <- tolower(activation)
  
  if (activation == "tanh") {
    return(1 - base::tanh(U)^2)
  }
  
  if (activation == "sigmoid") {
    S <- sigmoid_nominal(U)
    return(S * (1 - S))
  }
  
  if (activation == "relu") {
    return((U > 0) * 1)
  }
  
  stop("Unknown activation function, valid functions are tanh, sigmoid and relu.")
}


# Total number of parameters
theta_dimension_nominal <- function(
    p,
    neurons,
    K
) {
  q <- neurons * K +
    neurons +
    p * neurons +
    K
  
  list(q = q)
}


# Convert a list of paramters to a vector
flatten_theta_nominal <- function(theta) {
  c(
    as.vector(theta$W_out),
    theta$b_hidden,
    as.vector(theta$Lambda),
    theta$b_out
  )
}


# Obtains a list of paramters using a vector
unflatten_theta_nominal <- function(
    theta_vec,
    p,
    neurons,
    K
) {
  idx <- 1L
  
  W_out_len <- neurons * K
  
  W_out <- matrix(
    theta_vec[idx:(idx + W_out_len - 1L)],
    nrow = neurons,
    ncol = K
  )
  
  idx <- idx + W_out_len
  
  b_hidden <- theta_vec[
    idx:(idx + neurons - 1L)
  ]
  
  idx <- idx + neurons
  
  Lambda_len <- p * neurons
  
  Lambda <- matrix(
    theta_vec[idx:(idx + Lambda_len - 1L)],
    nrow = p,
    ncol = neurons
  )
  
  idx <- idx + Lambda_len
  
  b_out <- theta_vec[
    idx:(idx + K - 1L)
  ]
  
  list(
    W_out = W_out,
    b_hidden = b_hidden,
    Lambda = Lambda,
    b_out = b_out
  )
}

# Initializing parameters using the Nguyen-Widrow algorithm

init_theta_nominal <- function(
    neurons,
    p,
    K
) {
  W_out <- matrix(
    runif(
      neurons * K,
      min = -0.5,
      max = 0.5
    ),
    nrow = neurons,
    ncol = K
  )
  
  b_hidden <- runif(
    neurons,
    min = -0.5,
    max = 0.5
  )
  
  Lambda <- matrix(
    runif(
      p * neurons,
      min = -0.5,
      max = 0.5
    ),
    nrow = p,
    ncol = neurons
  )
  
  b_out <- runif(
    K,
    min = -0.5,
    max = 0.5
  )
  
  if (p == 1) {
    scaling_factor <- 0.7 * neurons
    
    for (k in seq_len(neurons)) {
      Lambda[, k] <- scaling_factor
      
      b_hidden[k] <- runif(
        1,
        -scaling_factor,
        scaling_factor
      )
    }
  } else {
    scaling_factor <- 0.7 * neurons^(1 / p)
    b <- seq(-1, 1, length.out = neurons)
    
    for (k in seq_len(neurons)) {
      norm_lambda <- sqrt(
        sum(Lambda[, k]^2)
      )
      
      if (norm_lambda == 0) {
        norm_lambda <- 1
      }
      
      Lambda[, k] <-
        scaling_factor *
        Lambda[, k] /
        norm_lambda
      
      b_hidden[k] <-
        scaling_factor *
        b[k] *
        sign(Lambda[1, k])
      
      if (b_hidden[k] == 0) {
        b_hidden[k] <-
          scaling_factor *
          b[k]
      }
    }
  }
  
  list(
    W_out = W_out,
    b_hidden = b_hidden,
    Lambda = Lambda,
    b_out = b_out
  )
}


# Selecting activation function using numeric code
# useful for C routines
activation_code_nominal <- function(
    activation = "tanh"
) {
  activation <- tolower(activation)
  
  code <- match(
    activation,
    c("tanh", "sigmoid", "relu")
  ) - 1L
  
  if (is.na(code)) {
    stop("Unknown activation function.")
  }
  
  return(code)
}

# Forward propagation using the function forward_pass_nominal in C

forward_pass_nominal_R <- function(
    X,
    theta,
    activation = "tanh"
) {
  X <- as.matrix(X)
  
  n <- nrow(X)
  p <- ncol(X)
  s <- ncol(theta$Lambda)
  K <- ncol(theta$W_out)
  
  out <- .Call(
    "forward_pass_nominal",
    
    as.double(X),
    as.integer(n),
    as.integer(p),
    
    as.double(theta$W_out),
    as.double(theta$b_hidden),
    as.double(theta$Lambda),
    as.double(theta$b_out),
    
    as.integer(s),
    as.integer(K),
    as.integer(
      activation_code_nominal(activation)
    ),
    
    double(n * s),
    double(n * s),
    double(n * K),
    double(n * K),
    
    PACKAGE = "brnn"
  )
  
  list(
    U = matrix(
      out$U,
      nrow = n,
      ncol = s
    ),
    
    H = matrix(
      out$H,
      nrow = n,
      ncol = s
    ),
    
    O = matrix(
      out$O,
      nrow = n,
      ncol = K
    ),
    
    P = matrix(
      out$P,
      nrow = n,
      ncol = K
    )
  )
}

# Sum of squares of the parameters
Ew_nominal <- function(theta) {
  sum(
    flatten_theta_nominal(theta)^2
  )
}

# Cross entropy with quadratic penalization
loss_function_nominal <- function(
    Y,
    P,
    theta,
    alpha,
    eps = 1e-10
) {
  L <- -sum(Y * log(P + eps))
  EW <- 0.5 * Ew_nominal(theta)
  F <- L + alpha * EW
  
  list(
    L = L,
    EW = EW,
    F = F
  )
}

# Compute the gradient for the objective function
gradient_function_nominal <- function(
    X,
    Y,
    theta,
    fp,
    activation = "tanh",
    alpha = 0
) {
  P <- fp$P
  U <- fp$U
  H <- fp$H
  
  delta_out <- P - Y
  
  # Gradient with respect to W_out
  grad_W_out <- t(H) %*% delta_out
  
  # Gradient with respect to b_out
  grad_b_out <- colSums(delta_out)
  
  # Signal to the hidden layer
  Gp <- activation_prime_matrix_nominal(
    U,
    activation
  )
  
  hidden_signal <-
    (delta_out %*% t(theta$W_out)) *
    Gp
 
  # Gradient with respect to b_hidden 
  grad_b_hidden <- colSums(
    hidden_signal
  )
  
  # Gradient with respect to Lambda
  grad_Lambda <- t(X) %*%
    hidden_signal
  
  grad_theta <- list(
    W_out = grad_W_out,
    b_hidden = grad_b_hidden,
    Lambda = grad_Lambda,
    b_out = grad_b_out
  )
  
  theta_vec <- flatten_theta_nominal(
    theta
  )
  
  grad_vec <-
    flatten_theta_nominal(grad_theta) +
    alpha * theta_vec
  
  list(
    grad_theta = unflatten_theta_nominal(
      grad_vec,
      p = ncol(X),
      neurons = ncol(theta$Lambda),
      K = length(theta$b_out)
    ),
    grad_vec = grad_vec
  )
}

# Fisher/Gauss-Newton approximation using the C 
# function hessian_approx_nominal
hessian_approx_nominal_R <- function(
    X,
    theta,
    fp,
    activation = "tanh"
) {
  X <- as.matrix(X)
  
  n <- nrow(X)
  p <- ncol(X)
  s <- ncol(theta$Lambda)
  K <- ncol(theta$W_out)
  
  q <- s * K +
    s +
    p * s +
    K
  
  out <- .Call(
    "hessian_approx_nominal",
    
    as.double(X),
    as.double(fp$P),
    as.double(fp$U),
    as.double(fp$H),
    
    as.double(theta$W_out),
    as.double(theta$Lambda),
    
    as.integer(n),
    as.integer(p),
    as.integer(s),
    as.integer(K),
    
    as.integer(
      activation_code_nominal(activation)
    ),
    
    double(q * q),
    
    PACKAGE = "brnn"
  )
  
  matrix(
    out,
    nrow = q,
    ncol = q
  )
}

# Solution of a linear system of equation using the
# Cholesky factorization
solve_chol_nominal <- function(A, b) {
  tryCatch({
    U <- chol(A)
    
    backsolve(
      U,
      backsolve(
        U,
        b,
        transpose = TRUE
      )
    )
  }, error = function(e) {
    NULL
  })
}

# Update the regularization parameter
update_alpha_nominal <- function(
    H_data,
    theta,
    alpha,
    alpha_min = 1e-8,
    alpha_max = 1e8
) {
  q <- nrow(H_data)
  EW_full <- 0.5 * Ew_nominal(theta)
  
  if (EW_full <= 0) {
    return(
      list(
        alpha = alpha,
        gamma = q
      )
    )
  }
  
  A_reg <- H_data
  diag(A_reg) <- diag(A_reg) + alpha
  
  trace_inv <- tryCatch({
    R <- chol(A_reg)
    sum(diag(chol2inv(R)))
  }, error = function(e) {
    NA_real_
  })
  
  if (!is.finite(trace_inv)) {
    return(
      list(
        alpha = alpha,
        gamma = q
      )
    )
  }
  
  gamma <- q - alpha * trace_inv
  gamma <- max(
    min(gamma, q),
    1e-8
  )
  
  alpha_new <- gamma / EW_full
  alpha_new <- max(
    min(alpha_new, alpha_max),
    alpha_min
  )
  
  list(
    alpha = alpha_new,
    gamma = gamma
  )
}

# Class membership probability
predict_proba_nominal <- function(
    object,
    newx
) {
  X <- as.matrix(newx)
  
  if (isTRUE(object$normalize)) {
    X <- normalize_nominal(
      X,
      base = object$x_base,
      spread = object$x_spread
    )
  }
  
  fp <- forward_pass_nominal_R(
    X,
    object$theta,
    object$activation
  )
  
  fp$P
}


# Prediction of class based on class membership probability
predict_class_nominal <- function(
    object,
    newx
) {
  P <- predict_proba_nominal(
    object,
    newx
  )
  
  idx <- max.col(
    P,
    ties.method = "first"
  )
  
  factor(
    object$classes[idx],
    levels = object$classes
  )
}

# S3 method for obtaining predictions for the nominal model
predict.brnn_nominal <- function(
    object,
    newdata = NULL,
    type = c("class", "prob"),
    ...
) {
  type <- match.arg(type)
  
  if (!inherits(object, "brnn_nominal")) {
    stop("This function only works for objects of class 'brnn_nominal'.")
  }
  
  dots <- list(...)
  
  if (length(dots) > 0L) {
    stop(
      "Unused argument(s): ",
      paste(names(dots), collapse = ", ")
    )
  }
  
  # If new data are not provided return fitted values for training data
  if (is.null(newdata)) {
    if (type == "class") {
      return(object$fitted_class)
    }
    
    return(object$fitted_prob)
  }
  
  # Fitted model using the formula interface
  if (inherits(object, "brnn_nominal.formula")) {
    newdata <- as.data.frame(newdata)
    
    Terms <- delete.response(object$terms)
    
    mf <- model.frame(
      Terms,
      newdata,
      na.action = na.omit,
      xlev = object$xlevels
    )
    
    if (!is.null(cl <- attr(Terms, "dataClasses"))) {
      .checkMFClasses(cl, mf)
    }
    
    X <- model.matrix(
      Terms,
      mf,
      contrasts.arg = object$contrasts
    )
    
    # The neural net already includes its own biases
    xint <- match(
      "(Intercept)",
      colnames(X),
      nomatch = 0L
    )
    
    if (xint > 0L) {
      X <- X[, -xint, drop = FALSE]
    }
  } else {
    # The model is fitted providing the information in matrices without the
    # formula interface
    if (is.null(dim(newdata))) {
      dim(newdata) <- c(1L, length(newdata))
    }
    
    X <- as.matrix(newdata)
    
    if (any(is.na(X))) {
      stop("Missing values are not allowed in 'newdata'.")
    }
  }
  
  if (ncol(X) != object$p) {
    stop(
      "'newdata' must contain ",
      object$p,
      " predictor variables."
    )
  }
  
  if (type == "prob") {
    return(
      predict_proba_nominal(
        object,
        X
      )
    )
  }
  
  predict_class_nominal(
    object,
    X
  )
}

# Fitting a neural network for nominal data
# Generic interface in R
brnn_nominal <- function(x, ...) {
  UseMethod("brnn_nominal")
}
# Metodo con formula para el ajuste nominal
brnn_nominal.formula <- function(
    formula,
    data,
    contrasts = NULL,
    ...
) {
  m <- match.call(expand.dots = FALSE)
  
  if (is.matrix(eval.parent(m$data))) {
    m$data <- as.data.frame(data)
  }
  
  m$... <- NULL
  m$contrasts <- NULL
  m[[1L]] <- as.name("model.frame")
  m <- eval.parent(m)
  
  Terms <- attr(m, "terms")
  
  x <- model.matrix(
    Terms,
    m,
    contrasts.arg = contrasts
  )
  
  cons <- attr(x, "contrasts")
  
  # The neural network already includes its own biases
  xint <- match(
    "(Intercept)",
    colnames(x),
    nomatch = 0L
  )
  
  if (xint > 0L) {
    x <- x[, -xint, drop = FALSE]
  }
  
  y <- model.response(m)
  
  out <- brnn_nominal.default(
    x = x,
    y = y,
    ...
  )
  
  out$terms <- Terms
  out$coefnames <- colnames(x)
  out$call <- match.call()
  out$contrasts <- cons
  out$xlevels <- .getXlevels(Terms, m)
  
  class(out) <- c(
    "brnn_nominal.formula",
    "brnn_nominal"
  )
  
  return(out)
}

# Default method for matrices or data.frames
brnn_nominal.default <- function(
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
    ...
) {
  
  # Initial checkings
  X <- as.matrix(x)
  
  if (!is.matrix(X)) stop("'x' must be convertible to a matrix.")
  if (nrow(X) < 2) stop("At least two observations are required.")
  if (length(y) != nrow(X)) {
    stop("The length of 'y' must match the number of rows in 'x'.")
  }
  
  y <- as.factor(y)
  enc <- one_hot_nominal(y)
  Y <- enc$Y
  classes <- enc$classes
  
  n <- nrow(X)
  p <- ncol(X)
  K <- ncol(Y)
  s <- neurons
  
  # Normalize predictors
  if (normalize) {
    x_base <- apply(X, 2, min)
    x_spread <- apply(X, 2, max) - x_base
    x_spread[x_spread == 0] <- 1
    
    Xn <- normalize_nominal(
      X,
      base = x_base,
      spread = x_spread
    )
  } else {
    Xn <- X
    x_base <- NULL
    x_spread <- NULL
  }
  
  # Parameter initialization
  set.seed(seed)
  
  theta <- init_theta_nominal(
    neurons = s,
    p = p,
    K = K
  )
  
  alpha <- alpha_init
  
  q <- theta_dimension_nominal(
    p,
    s,
    K
  )$q
  
  if (verbose) {
    cat("------------------------------------------------------------\n")
    cat("Bayesian regularized neural network for nominal classification\n")
    cat("Observations:", n, "\n")
    cat("Predictors:", p, "\n")
    cat("Classes:", K, "\n")
    cat("Hidden neurons:", s, "\n")
    cat("Total number of parameters to estimate:", q, "\n")
    cat("Activation:", activation, "\n")
    cat("Nguyen-Widrow initialization\n")
    cat("------------------------------------------------------------\n")
  }
  
  # Training history
  F_history <- numeric(epochs)
  L_history <- numeric(epochs)
  Ew_history <- numeric(epochs)
  alpha_history <- numeric(epochs)
  gamma_history <- numeric(epochs)
  mu_history <- numeric(epochs)
  grad_history <- numeric(epochs)
  
  reason <- "UNKNOWN"
  epoch <- 1
  converged <- FALSE
  gamma <- NA_real_
  
  while (epoch <= epochs) {
    
    # Forward propagation and evaluation of the objective function
    fp <- forward_pass_nominal_R(
      Xn,
      theta,
      activation
    )
    
    loss_now <- loss_function_nominal(
      Y,
      fp$P,
      theta,
      alpha
    )
    
    # Gradient
    gr_data <- gradient_function_nominal(
      X = Xn,
      Y = Y,
      theta = theta,
      fp = fp,
      activation = activation,
      alpha = 0
    )
    
    grad_data <- gr_data$grad_vec
    
    theta_vec <- flatten_theta_nominal(
      theta
    )
    
    grad <- grad_data + alpha * theta_vec
    grad_norm <- sqrt(sum(grad^2))
    
    # Approximation of the Hessian matrix
    H_data <- hessian_approx_nominal_R(
      X = Xn,
      theta = theta,
      fp = fp,
      activation = activation
    )
    
    # Update alpha
    upd_alpha <- update_alpha_nominal(
      H_data,
      theta,
      alpha
    )
    
    alpha <- upd_alpha$alpha
    gamma <- upd_alpha$gamma
    
    # Evaluation using the updated alpha
    loss_now <- loss_function_nominal(
      Y,
      fp$P,
      theta,
      alpha
    )
    
    grad <- grad_data + alpha * theta_vec
    grad_norm <- sqrt(sum(grad^2))
    F_now <- loss_now$F
    
    F_history[epoch] <- F_now
    L_history[epoch] <- loss_now$L
    Ew_history[epoch] <- loss_now$EW
    alpha_history[epoch] <- alpha
    gamma_history[epoch] <- gamma
    mu_history[epoch] <- mu
    grad_history[epoch] <- grad_norm
    
    if (verbose) {
      cat(
        "Epoch =", epoch,
        "| F =", round(F_now, 8),
        "| L =", round(loss_now$L, 8),
        "| Ew =", round(loss_now$EW, 8),
        "| alpha =", format(alpha, scientific = TRUE),
        "| gamma =", round(gamma, 6),
        "| grad =", round(grad_norm, 8),
        "| mu =", format(mu, scientific = TRUE),
        "\n"
      )
    }
    
    # Stoping criteria using the norm of the gradient
    if (grad_norm < min_grad) {
      converged <- TRUE
      reason <- "Gradient norm less than min_grad"
      break
    }
    
    # Levenberg-Marquardt step
    accepted <- FALSE
    
    while (!accepted && mu <= mu_max) {
      A <- H_data
      diag(A) <- diag(A) + alpha + mu
      
      step <- solve_chol_nominal(
        A,
        -grad
      )
      
      if (is.null(step)) {
        mu <- mu * mu_inc
        next
      }
      
      theta_vec <- flatten_theta_nominal(
        theta
      )
      
      theta_prop_vec <-
        theta_vec +
        as.vector(step)
      
      theta_prop <- unflatten_theta_nominal(
        theta_prop_vec,
        p = p,
        neurons = s,
        K = K
      )
      
      fp_prop <- forward_pass_nominal_R(
        Xn,
        theta_prop,
        activation
      )
      
      loss_prop <- loss_function_nominal(
        Y,
        fp_prop$P,
        theta_prop,
        alpha
      )
      
      F_prop <- loss_prop$F
      
      if (
        is.finite(F_prop) &&
        (F_prop < F_now)
      ) {
        theta <- theta_prop
        mu <- max(
          mu * mu_dec,
          1e-20
        )
        accepted <- TRUE
      } else {
        mu <- mu * mu_inc
      }
    }
    
    if (!accepted) {
      reason <- "No acceptable step found or mu_max reached"
      break
    }
    
    # Criterio based on changes of objective function
    if (epoch >= 2) {
      if (
        abs(
          F_history[epoch] -
          F_history[epoch - 1]
        ) < change
      ) {
        converged <- TRUE
        reason <- "Absolute change in F less than change"
        break
      }
    }
    
    epoch <- epoch + 1
  }
  
  if (epoch > epochs) {
    reason <- "Maximum number of epochs reached"
    epoch <- epochs
  }
  
  # Trim the training history
  F_history <- F_history[1:epoch]
  L_history <- L_history[1:epoch]
  Ew_history <- Ew_history[1:epoch]
  alpha_history <- alpha_history[1:epoch]
  gamma_history <- gamma_history[1:epoch]
  mu_history <- mu_history[1:epoch]
  grad_history <- grad_history[1:epoch]
  
  # Final pass for fitting the model
  fp_final <- forward_pass_nominal_R(
    Xn,
    theta,
    activation
  )
  
  loss_final <- loss_function_nominal(
    Y,
    fp_final$P,
    theta,
    alpha
  )
  
  fitted_class <- factor(
    classes[
      max.col(
        fp_final$P,
        ties.method = "first"
      )
    ],
    levels = classes
  )
  
  acc <- mean(fitted_class == y)
  
  if (verbose) {
    cat("------------------------------------------------------------\n")
    cat("Training completed\n")
    cat("Reason:", reason, "\n")
    cat("Epochs used:", epoch, "\n")
    cat("Final F:", round(loss_final$F, 8), "\n")
    cat("Final alpha:", format(alpha, scientific = TRUE), "\n")
    cat("Training accuracy:", round(acc, 6), "\n")
    cat("------------------------------------------------------------\n")
  }
  
  out <- list(
    theta = theta,
    alpha = alpha,
    gamma = gamma_history[length(gamma_history)],
    epochs_used = epoch,
    reason = reason,
    converged = converged,
    activation = activation,
    neurons = s,
    n = n,
    p = p,
    K = K,
    q = q,
    classes = classes,
    coefnames = if (!is.null(colnames(X))) {
      colnames(X)
    } else {
      paste0("x", seq_len(p))
    },
    fitted_prob = fp_final$P,
    fitted_class = fitted_class,
    y = y,
    F_history = F_history,
    L_history = L_history,
    Ew_history = Ew_history,
    alpha_history = alpha_history,
    gamma_history = gamma_history,
    mu_history = mu_history,
    grad_history = grad_history,
    normalize = normalize,
    x_base = x_base,
    x_spread = x_spread
  )
  
  class(out) <- "brnn_nominal"
  
  return(out)
}

# print method for models with nominal respones
print.brnn_nominal <- function(x, ...) {
  if (!inherits(x, "brnn_nominal")) {
    stop("This function only works for objects of class 'brnn_nominal'.")
  }
  
  cat("A Bayesian regularized neural network for nominal classification\n")
  
  cat(
    x$p, "-", x$neurons, "-", x$K,
    " network with ", x$q,
    " weights and biases\n",
    sep = ""
  )
  
  cat(
    "Inputs were ",
    if (isTRUE(x$normalize)) "" else "not ",
    "normalized\n",
    sep = ""
  )
  
  cat(
    "Classes: ",
    paste(x$classes, collapse = ", "),
    "\n",
    sep = ""
  )
  
  cat(
    "Training finished because ",
    x$reason,
    "\n",
    sep = ""
  )
  
  invisible(x)
}

# Method to extract the parameters for the nominal model

coef.brnn_nominal <- function(object, ...) {
  if (!inherits(object, "brnn_nominal")) {
    stop(
      "This function only works for objects of class 'brnn_nominal'."
    )
  }
  
  theta <- object$theta
  
  predictor_names <- object$coefnames
  
  if (
    is.null(predictor_names) ||
    length(predictor_names) != object$p
  ) {
    predictor_names <- paste0(
      "x",
      seq_len(object$p)
    )
  }
  
  # Weights from the hidden layer to the classes
  W_out_names <- unlist(
    lapply(
      seq_len(object$K),
      function(k) {
        paste0(
          "W_out[",
          seq_len(object$neurons),
          ",",
          object$classes[k],
          "]"
        )
      }
    ),
    use.names = FALSE
  )
  
  # Biases for the neurons in the hidden layer
  b_hidden_names <- paste0(
    "b_hidden[",
    seq_len(object$neurons),
    "]"
  )
  
  # Weights of the covariates to the hidden layer
  Lambda_names <- unlist(
    lapply(
      seq_len(object$neurons),
      function(h) {
        paste0(
          "Lambda[",
          predictor_names,
          ",",
          h,
          "]"
        )
      }
    ),
    use.names = FALSE
  )
  
  # Biases associated to each class in the output layer
  b_out_names <- paste0(
    "b_out[",
    object$classes,
    "]"
  )
  
  coefficients <- flatten_theta_nominal(theta)
  
  names(coefficients) <- c(
    W_out_names,
    b_hidden_names,
    Lambda_names,
    b_out_names
  )
  
  coefficients
}

## Model summary 


summary.brnn_nominal <- function(object, ...) {
  cat("------------------------------------------------------------\n")
  cat("Model summary\n")
  cat("Classes:", paste(object$classes, collapse = ", "), "\n")
  cat("Observations:", object$n, "\n")
  cat("Predictors:", object$p, "\n")
  cat("Hidden neurons:", object$neurons, "\n")
  cat("Parameters:", object$q, "\n")
  cat("Activation:", object$activation, "\n")
  cat("Final alpha:", object$alpha, "\n")
  cat("Final gamma:", object$gamma, "\n")
  cat("Epochs used:", object$epochs_used, "\n")
  cat("Converged:", object$converged, "\n")
  cat("Reason:", object$reason, "\n")
  cat(
    "Training accuracy:",
    mean(object$fitted_class == object$y),
    "\n"
  )
  cat("------------------------------------------------------------\n")
  
  invisible(object)
}

## Confusion matrix 

# Confusion matrix for observed and predicted classes
confusion_brnn_nominal <- function(
    object,
    y_true = NULL,
    y_pred = NULL
) {
  if (is.null(y_true)) {
    y_true <- object$y
  }
  
  if (is.null(y_pred)) {
    y_pred <- object$fitted_class
  }
  
  table(
    Observed = y_true,
    Predicted = y_pred
  )
}
