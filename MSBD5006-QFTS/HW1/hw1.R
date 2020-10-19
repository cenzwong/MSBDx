setwd("~/R-Script/HW1")

# Data prepare
cpileng = read.table("m-cpileng.txt",header=F) 
dec19 = read.table("m-dec19.txt",header=T) 
gnprate = read.table("q-gnprate.txt",header=F) 

# Q1
# plot(dec19["dec1"],main="Nasdaq daily simple return",ylab="",xlab="",type="l")
dec19_dec1 = dec19["dec1"]
dec19_ACF = acf(dec19_dec1,24)
dec19_PACF = pacf(dec19_dec1,24)

#Test mean being zero.
Box.test(dec19_dec1,lag=12,type="Ljung")

# Q2
# plot(dec19["dec1"],main="Nasdaq daily simple return",ylab="",xlab="",type="l")
dec19_dec9 = dec19["dec9"]
dec19_ACF = acf(dec19_dec9,12)
dec19_PACF = pacf(dec19_dec9,12)

#Test mean being zero.
Box.test(dec19_dec9,lag=12,type="Ljung")

# Q3 (I don't know how to shift in R)
cpileng_CPI = cpileng[4]
C_t = log(cpileng_CPI) - log(lag(cpileng_CPI, lag = 1))

cpileng_CPI_ACF = acf(cpileng_CPI,12)
cpileng_CPI_PACF = pacf(cpileng_CPI,12)

#Test mean being zero.
Box.test(cpileng_CPI,lag=12,type="Ljung")
