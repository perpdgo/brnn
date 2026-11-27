##############################################################
#Example 11
#Gianola et al. (2011).
#Warning, it will take a while, substitute the FALSE
#statement with the TRUE statement if you really want to run the example

#This example uses the formula interface for the fitting

#Load the Jersey dataset
data(Jersey)

# using y~\Gamma*X model
descomp = eigen(G, symmetric = TRUE)
Gamma = descomp$vectors
Lambda = descomp$values

indice=Lambda>1e-6
Gamma=Gamma[,indice]
Lambda = Lambda[indice]

X = Gamma %*% diag(sqrt(Lambda))
  
#Fit the model with the FULL DATA
#Formula interface
out=brnn(pheno$yield_devMilk~X,neurons=2,verbose=TRUE)
  
#Obtain predictions and plot them against fitted values
plot(pheno$yield_devMilk,predict(out))
