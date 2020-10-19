setwd("~/R-Script/week-1/data")
# set the working directory

da=read.csv("Nasdaq_daily.csv",header=T) 
#header=T means 1st row of the data file contains variable names. The default is header=F, i.e., no names.

# This is the Close value
B=da[,5]
#See the first column of the data

A=rev(B)
ndaily=length(A)
ndaily

#Compute the simple return
# A[2:ndaily] is the t-1 dataset
# A[1:(ndaily-1)] is the t dataset
s_rtn=(A[2:ndaily]/A[1:(ndaily-1)]-1)*100

#Compute the log return 
l_rtn=diff(log(A))*100 

#s_rtn=exp(l_rtn)-1
plot(A,main="Nasdaq daily closing price",ylab="",xlab="",type="l")
plot(s_rtn,main="Nasdaq daily simple return",ylab="",xlab="",type="l")
plot(l_rtn,main="Nasdaq daily log return",ylab="",xlab="",type="l")


# Compute the summary statistics
library(fBasics) # Load the package fBasics.
basicStats(s_rtn) 

#Alternatively, one can use individual commands as follows:
mean(s_rtn)
var(s_rtn)
sqrt(var(s_rtn)) # Standard deviation
skewness(s_rtn)
kurtosis(s_rtn)

#Simple tests
n=length(s_rtn)
s1=skewness(s_rtn)
t1=s1/sqrt(6/n) #Compute test statistic
t1

pv=2*pnorm(t1) #Compute p-value.
pv


#Test mean being zero.
t.test(s_rtn) 
#The result shows that the hypothesis of zero expected return can be rejected at the 5%  level.
###
# One Sample t-test
# 
# data:  s_rtn
# t = 3.2707, df = 9079, p-value = 0.001077
# alternative hypothesis: true mean is not equal to 0
# 95 percent confidence interval:
#   0.01839958 0.07344426
# sample estimates:
#   mean of x 
# 0.04592192 
###


#Normality test
normalTest(s_rtn,method="jb") 
#The result shows the normality for simple return is rejected

##############################################################
C=read.csv("Nasdaq_monthly.csv", header=T)[,5]
A=rev(C)
ndaily=length(A)

s_rtn=(A[2:ndaily]/A[1:(ndaily-1)]-1)*100
l_rtn=diff(log(A))*100 

#s_rtn=exp(l_rtn)-1
plot(A,main="Nasdaq monthly closing price",ylab="",xlab="",type="l")
plot(s_rtn,main="Nasdaq monthly simple return",ylab="",xlab="",type="l")
plot(l_rtn,main="Nasdaq monthly log return",ylab="",xlab="",type="l")



#Study the emprical density
C=read.csv("Nasdaq_monthly.csv",header=T)[,5]
A=rev(C)
ndaily=length(A)
s_rtn=A[2:ndaily]/A[1:(ndaily-1)]-1
l_rtn=diff(log(A))
 

plot(density(l_rtn),lty=1,lwd=1.2,col="red",xlim=c(-0.4,0.4),ylim=c(0,8.2),main="",ylab="Density",xlab="")
x=seq(-0.4,0.4,0.01)
lines(x,dnorm(x,mean(l_rtn),sd(l_rtn)),lty=2,lwd=1.2,,col="green3",xlim=c(-0.4,0.4),ylim=c(0,8.2))
legend("topright",c("Log return", "Normal"),lty=c(1,2),lwd=1.2,col=c("red","green3"),cex=0.6)

plot(density(s_rtn),lty=1,lwd=1.2,col="red",xlim=c(-0.4,0.4),ylim=c(0,8.2),main="",ylab="Density",xlab="")
x=seq(-0.4,0.4,0.01)
lines(x,dnorm(x,mean(l_rtn),sd(l_rtn)),lty=2,lwd=1.2,,col="green3",xlim=c(-0.4,0.4),ylim=c(0,8.2))
legend("topright",c("S return", "Normal"),lty=c(1,2),lwd=1.2,col=c("red","green3"),cex=0.6)






# s=(l_rtn-mean(l_rtn))/sd(l_rtn)
# plot(density(s),lty=1,lwd=1.2,col="red",xlim=c(-6,6),ylim=c(0,0.75),main="",ylab="",xlab="l_rtn")
# x=seq(-5,5,0.1)
# lines(x,dnorm(x),lty=2,lwd=1.2,,col="green3",xlim=c(-6,6),ylim=c(0,0.5))
# legend("topright",legend=c("l_rtn",expression(N(0,1))),lty=c(1,2),lwd=1.2,col=c("red","green3")
#        ,cex=0.6)
#########################################

