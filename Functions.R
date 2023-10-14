require("foreign")
require("caret")

#Predict the labels for a given test set (x.test) based on training data (x.train) 
#and the corresponding training labels (y.train)
knnPredictions <- function(x.train, x.test, y.train){
  if(nrow(x.train) != length(y.train)){
    stop("Length of training labels must be equal to number of rows")  
  }
  ntrain <- nrow(x.train)
  ntest <- nrow(x.test)
  y.pred <- vector()
  
  for(i in 1:ntest){
    tmp <- x.test[i,]
    dists <- vector()
    for(j in 1:ntrain){
      dists[j] <- TODO
    }
    min.dist <- min(dists)
    y.pred[i] <- y.train[dists==min.dist][1]
  }
  return(y.pred)
}


#calculates the euclidean distance between two instances/objects. 
euclidDist <- function(o1,o2){
  TODO
}

#calculates the accuracy using the predicted (y.predict) and actual labels (y.test)
acc <- function(y.predict, y.test){
  conf.mat <- table(y.predict , y.test)
  TODO
}




