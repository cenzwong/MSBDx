# Simple return, (now - shift)/shift * 100
s_rtn=(A[2:ndaily]/A[1:(ndaily-1)]-1)*100

#Compute the log return 
l_rtn=diff(log(A))*100 

#s_rtn=exp(l_rtn)-1
plot(A,main="Nasdaq daily closing price",ylab="",xlab="",type="l")
plot(s_rtn,main="Nasdaq daily simple return",ylab="",xlab="",type="l")
plot(l_rtn,main="Nasdaq daily log return",ylab="",xlab="",type="l")

# To find those statistic
# Compute the summary statistics
library(fBasics) # Load the package fBasics.
basicStats(s_rtn) 

#Alternatively, one can use individual commands as follows:
mean(s_rtn)
var(s_rtn)
sqrt(var(s_rtn)) # Standard deviation
skewness(s_rtn)
kurtosis(s_rtn)
