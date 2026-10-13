# Example 7
# Nominal response classification

# Load the library
library(brnn)

# Training and testing sets
set.seed(123)

training = sample(
  1:nrow(iris),
  size = floor(0.80 * nrow(iris)),
  replace = FALSE
)

irisTrain = iris[training, ]
irisTest = iris[-training, ]

# Fit a nominal Bayesian regularized neural network
fmNominal = brnn_nominal(
  Species ~ .,
  data = irisTrain,
  neurons = 2,
  activation = "tanh",
  normalize = TRUE,
  epochs = 100,
  verbose = FALSE,
  seed = 123
)

# Model summary
summary(fmNominal)

# Predicted classes
predictedClass = predict(
  fmNominal,
  newdata = irisTest,
  type = "class"
)

# Predicted class probabilities
predictedProbability = predict(
  fmNominal,
  newdata = irisTest,
  type = "prob"
)

# Confusion matrix
confusionMatrix = table(
  Observed = irisTest$Species,
  Predicted = predictedClass
)

confusionMatrix

# Classification accuracy
mean(predictedClass == irisTest$Species)

# First predicted probabilities
head(predictedProbability)