
library(seqknockoff)
set.seed(123)
xtrain =read.csv("./xtrain.csv")[,-1]
# X3, X4, X5 is floats, other variables are binary/cateogorical 
xtrain[,!names(xtrain) %in% c("X3", "X4", "X5")] = lapply(xtrain[,!names(xtrain) %in% c("X3", "X4", "X5")], as.factor)
xtrain[,names(xtrain) %in% c("X3", "X4", "X5")] = lapply(xtrain[,names(xtrain) %in% c("X3", "X4", "X5")], as.numeric)
xtrain_ko = seqknockoff::knockoffs_seq(xtrain[, ! names(xtrain) %in% c("X19")]) # exclude X19 because it is all 0's -- add 0's later on again

xtrain_ko_final = cbind(xtrain_ko[,1:19], "X19" = 0,xtrain_ko[,20:27] )
write.csv(as.data.frame(xtrain_ko_final), "./xtrain_ko.csv", row.names = F)

## generate multiple knockoffs for each instance, e.g. 10
set.seed(123)
xtest =read.csv("./xtest.csv")[,-1]
# drop single level
xtest = xtest[which(xtest$X8 <= 4.54),]
# X3, X4, X5 is floats, other variables are binary/cateogorical 
xtest[,!names(xtest) %in% c("X3", "X4", "X5")] = lapply(xtest[,!names(xtest) %in% c("X3", "X4", "X5")], as.factor)
xtest[,names(xtest) %in% c("X3", "X4", "X5")] = lapply(xtest[,names(xtest) %in% c("X3", "X4", "X5")], as.numeric)

rr = replicate(10, { xtest_ko = seqknockoff::knockoffs_seq(xtest[, ! names(xtest) %in% c("X19")])
cbind(xtest_ko[,1:19], "X19" = 0,xtest_ko[,20:27] )}, simplify = F)
write.csv(do.call(rbind,rr), "./xtest_10_ko.csv", row.names = F)
