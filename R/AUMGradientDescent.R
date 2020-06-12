
#import libraries and sources
library(data.table)
library(sigmoid)
library(MLmetrics)
library(ggplot2)
library(penaltyLearning)
source("AUMcalcAreas.R")
source("ROChange.R")

#import spam data set
import.data.list <- fread('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data')

#change imported data to matrix
data.mat <- as.matrix(import.data.list)

#separate data into X matrix and y labels
X <- scale(data.mat[,1:57])
y <- data.mat[,58]

#create train and validation fold vecs
fold.vec <- sample(rep(1:5, ceiling(nrow(X) / 5)), nrow(X))

#create validation data
X.val <- X[fold.vec == 1,]
y.val <- y[fold.vec == 1]

#create training data
X.train <- X[fold.vec != 1,]
y.train <- y[fold.vec != 1]

#parameter initialization
step.size <- 5
max.iter <- 10
weight.vec <- rnorm(dim(X)[2])

#print baseline accuracy
baseline.acc <- max(sum(y == 1)/length(y), sum(y==0)/length(y))
print(c("baseline accuracy", baseline.acc))

#initialize accuracy/metric variables
train.acc <- numeric()
val.acc <- numeric()
train.AUM.vec <- numeric()
train.AUC.vec <- numeric()
train.loss <- numeric()
val.AUM.vec <- numeric()
val.AUC.vec <- numeric()
val.loss <- numeric()

#full gradient descent
calcGrad <- function(X, y, weight.vec, lo, hi)
{
  #find gradient for every observation
  grad <- X *
          (as.vector(pmin(lo, hi)) *
          as.vector(exp( -1 * X %*% weight.vec)/(1 + exp(-1 * X %*% weight.vec)^2)))
  
  #find mean gradient
  grad <- as.vector(apply(grad, 2, mean))
  
  #print if we are at a nondifferentiable point
  if(lo != hi)
  {
    print("lo != hi")
  }
  
  #return gradient
  grad
}


start.time <- Sys.time()
#AUM gradient descent
for( curr.iter in 1:max.iter )
{
  print(curr.iter)
  
  #calculate predicted y value
  pred.y.value <- as.vector(sigmoid(X.train %*% weight.vec))
  
  #calculate AUM-gradient related metrics
  train.metrics <- calcAreas(pred.y.value, y.train)
  train.AUM.vec[curr.iter] <- train.metrics[[1]]
  lo <- train.metrics[[3]]
  hi <- train.metrics[[4]]
  
  #store AUC for comparative analytic purposes
  train.AUC.vec[curr.iter] <- train.metrics[[2]]
  
  #store log loss for comparative analyitic purposes
  train.loss[curr.iter] <- LogLoss(pred.y.value, y.train)
  
  #calculate validation predicted value
  val.pred.y.value <- as.vector(sigmoid(X.val %*% weight.vec))
  #calculate validation metrics
  val.metrics <- calcAreas(val.pred.y.value, y.val)
  val.AUM.vec[curr.iter] <- val.metrics[[1]]
  val.AUC.vec[curr.iter] <- val.metrics[[2]]
  val.loss[curr.iter] <- LogLoss(val.pred.y.value, y.val)
  
  #calculate gradient and update weights
  grad <- calcGrad(X.train, y.train, weight.vec, lo, hi)
  weight.vec <- weight.vec - (step.size * grad)
  
  #calculate predicted labels for accuracy calculation
  #Note that since this is penalty learning, higher predicted values are mapped
  #to negative labels
  pred.y.label <- ifelse(as.numeric(sigmoid(X.train %*% weight.vec)) > 0.5, 0, 1)
  val.pred.y.label <- ifelse(as.numeric(sigmoid(X.val %*% weight.vec)) > 0.5, 0, 1)
  
  #store accuracy
  train.acc[curr.iter] <- sum(pred.y.label == y.train) /length(y.train)
  val.acc[curr.iter] <- sum(val.pred.y.label == y.val) / length(y.val)
  
  #diagnostic check for divergence
  if(sum(is.na(train.acc)) != 0)
  {
    print("diverging!! help!!")
  }
}
#calculate the end.time for the training algorithm
end.time <- Sys.time()

#display how long the descent took
print(end.time - start.time)

#scaled the AUM so it fits properly
train.AUM.scaled <- train.AUM.vec / max(train.AUM.vec)
val.AUM.scaled <- val.AUM.vec / max(val.AUM.vec)

#comment out some rows so its not a spaghetti chart
dt <- data.table(Iteration = 1:max.iter, 
                 AUM = train.AUM.scaled, 
                 AUC = train.AUC.vec, 
                 Accuracy = train.acc, 
                 val.Accuracy = val.acc,
                 `Negative Loss` = (-train.loss - range(-train.loss)[1] ) / diff(range(-train.loss)),
                 val.AUM = val.AUM.scaled,
                 val.AUC = val.AUC.vec,
                 `Negative Val Loss` = (-val.loss - range(-val.loss)[1] ) / diff(range(-val.loss))
                 )

#melt data table so you can plot multiple lines
melted.dt <- melt(dt, id = "Iteration")
names(melted.dt) <- c("Iteration", "Metric", "Value")


ggplot()+
  geom_line(data = melted.dt, aes(x = Iteration, y = Value, color = Metric))

  # geom_point(data = dt, aes(x = which.max(Accuracy), y = max(Accuracy))) +
  # geom_point(data = dt, aes(x = which.max(val.Accuracy), y = max(val.Accuracy))) +
  # geom_point(data = dt, aes(x = which.max(AUC), y = max(AUC))) + 
  # geom_point(data = dt, aes(x = which.max(val.AUC), y = max(val.AUC))) + 
  # geom_point(data = dt, aes(x = which.min(AUM), y = min(AUM))) +
  # geom_point(data = dt, aes(x = which.min(val.AUM), y = min(val.AUM))) +
  # geom_point(data = dt, aes(x = which.min(`Negative Loss`), y = min(`Negative Loss`))) +
  # geom_point(data = dt, aes(x = which.min(`Negative Val Loss`), y = min(`Negative Val Loss`)))



