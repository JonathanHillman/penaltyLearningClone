#gradient helper function
#Input:
#y.pred: predicted values (not labels)
#y.true: true labels
#Outputs: a list contining the aum, auc, lo, and hi respectively
calcAreas <- function(y.pred, y.true)
{
  #duplicate each element of y.ture
  dup.y.true <- unlist(lapply(y.true, function(x){c(x,x)}))
  
  #each instance of dup.y.true = 0 implies the existence of a possible false positive
  possible.fp <- numeric(length(dup.y.true))
  possible.fp[dup.y.true == 0] = 1
  
  #each instance of dup.y.true = 1 implies the existence of a possible false negative
  possible.fn <- numeric(length(dup.y.true))
  possible.fn[dup.y.true == 1] = 1
  
  #create false positive vector based on y.true and penalty thresholds
  fp <- numeric(length(dup.y.true))
  fp <- unlist(lapply(y.true, function(x){c(ifelse(x==1, 0, 1), 0)}))
  
  #create false negative vector based on y.true and penalty thresholds
  fn <- numeric(length(dup.y.true))
  fn <- unlist(lapply(y.true, function(x){c(0, x)}))
  
  #create model data table for use in ROChange
  models <- data.table(
    fp = fp,
    fn = fn,
    possible.fn = possible.fn,
    possible.fp = possible.fp,
    min.log.lambda = rep( c(-Inf, 0.5), length(y.true)),
    max.log.lambda = rep( c(0.5, Inf), length(y.true)),
    labels = 1,
    problem = (unlist(lapply(1:length(y.true), function(x){c(x,x)})))
  )
  models[, errors := fp + fn]
  
  #create prediction data table for use in ROChange
  predictions <- data.table( problem = 1:length(y.pred), pred.log.lambda = y.pred)
  
  #store output of ROChange function
  L <- ROChange(models, predictions, "problem")
  
  #return required metrics for AUM-based gradient descents and analytics
  list(L$aum, L$auc, L$aum.grad$lo, L$aum.grad$hi)
}
