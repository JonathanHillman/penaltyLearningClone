
#import libraries
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

#parameter initialization
step.size <- 0.005
max.iter <- 300
weight.vec <- rnorm(dim(X)[2])

#accuracy initiliation
baseline.acc <- max(sum(y == 1)/length(y), sum(y==0)/length(y))
print(c("baseline accuracy", baseline.acc))
acc <- numeric()
AUM.vec <- numeric()
AUC.vec <- numeric()
loss <- numeric()

start.time <- Sys.time()
#Traditional Gradient Descent
for( curr.iter in 1:max.iter )
{
  #calculate output of learning function
  pred.y.value <- as.numeric(sigmoid(X %*% weight.vec))
  
  #calculate gradient and update weights
  grad <- apply((X * (pred.y.value - y)), 2, mean)#this is standard logistic regression
  weight.vec <- weight.vec + (step.size * grad)
  
  #calculate new predictions
  pred.y.label <- ifelse(as.numeric(sigmoid(X %*% weight.vec)) > 0.5, 0, 1)
  
  #print current iteration for diagnostic purposes
  #print(c("Current Traditional Gradient Iteration:",curr.iter))
  
  if( curr.iter %% 100 == 0)
  {
    metrics <- calcAreas(pred.y.value, y)
    AUM.vec[curr.iter] <- metrics[[1]]
    AUC.vec[curr.iter] <- metrics[[2]]
    print(c("Current AUM for iteration:", curr.iter, AUM.vec[curr.iter]))
  }
  
  
  #store accuracy
  acc[curr.iter] <- sum(pred.y.label == y) /length(y)
  loss[curr.iter] <- LogLoss(pred.y.value, y)
}
end.time <- Sys.time()

AUM.scaled <- (AUM.vec) / max(na.omit(AUM.vec))

dt <- na.omit(data.table(Iteration = 1:max.iter, AUM = AUM.scaled, AUC = AUC.vec, Accuracy = acc, `Negative Loss` = (-loss - range(-loss)[1] ) / diff(range(-loss))))
melted.dt <- melt(dt, id = "Iteration")
names(melted.dt) <- c("Iteration", "Metric", "Value")


ggplot()+
  geom_line(data = melted.dt, aes(x = Iteration, y = Value, color = Metric)) +
  geom_point(data = dt, aes(x = Iteration[which.max(Accuracy)], y = max(Accuracy))) +
  geom_point(data = dt, aes(x = Iteration[which.max(AUC)], y = max(AUC))) + 
  geom_point(data = dt, aes(x = Iteration[which.min(AUM)], y = min(AUM))) +
  geom_point(data = dt, aes(x = Iteration[which.min(`Negative Loss`)], y = min(`Negative Loss`)))






dt <- na.omit(data.table( 1:max.iter, baseline.acc, acc, AUM.vec, (AUM.vec / max(na.omit(AUM.vec))), AUC.vec ))
names(dt) <- c("Iteration","Baseline", "Train Accuracy","AUM", "AUM_scaled","AUC")

p <- ggplot( data = dt, aes(x = Iteration)) +
  geom_line(aes(y = baseline.acc), color = "blue") +
  geom_line(aes(y = `Train Accuracy`), color = "black") +
  geom_point(aes(x = dt$Iteration[which.max(dt$`Train Accuracy`)], y = max(`Train Accuracy`))) +
  geom_line(aes(y = AUM_scaled), color = "purple") +
  geom_point(aes(x = dt$Iteration[which.min(dt$AUM_scaled)], y = min(AUM_scaled))) +
  geom_line(aes(y = AUC), color = "red") +
  geom_point(aes(x = dt$Iteration[which.max(dt$AUC)], y = max(AUC)))

print(p)
print(c("traditional approach accuracy:", acc[max.iter]))
print(end.time - start.time)
