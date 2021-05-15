#Example 6
#Warning, it will take a while

#Load the library
library(brnn)

#Load the dataset
data(GLS)

#Subset of data for location Harare
HarareOrd=subset(phenoOrd,Loc=="Harare")

#Eigen value decomposition for GOrdm keep those 
#eigen vectors whose corresponding eigen-vectors are bigger than 1e-10
#and then compute principal components

evd=eigen(GOrd)
evd$vectors=evd$vectors[,evd$value>1e-10]
evd$values=evd$values[evd$values>1e-10]
PC=evd$vectors%*%sqrt(diag(evd$values))
rownames(PC)=rownames(GOrd)

#Response variable
y=phenoOrd$rating
gid=as.character(phenoOrd$Stock)

Z=model.matrix(~gid-1)
colnames(Z)=gsub("gid","",colnames(Z))

if(any(colnames(Z)!=rownames(PC))) stop("Ordering problem\n")

#Matrix of predictors for Neural net
X=Z%*%PC

#Cross-validation
set.seed(1)
testing=sample(1:length(y),size=as.integer(0.10*length(y)),replace=FALSE)
isNa=(1:length(y)%in%testing)
yTrain=y[!isNa]
XTrain=X[!isNa,]
nTest=sum(isNa)

neurons=2
	
fmOrd=brnn_ordinal(XTrain,yTrain,neurons=neurons,verbose=FALSE)

#Predictions for testing set
XTest=X[isNa,]
predictions=predict(fmOrd,XTest)
predictions
