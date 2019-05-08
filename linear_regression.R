#Performing exploratory Data Analysis
pairs(mtcars, main = "mtcars data", pch=16) #Plot all the variables against each other with a scatterplot matrix 
pairs(subset(mtcars, select=c("mpg", "wt", "disp")), main = "mtcars data", pch=16) #Plot those variables that seem to have a linear correlation


#Multiple variable linear regression using analytical solution 

y <- data.matrix(subset(mtcars, select=c("mpg")))

x <- data.matrix(subset(mtcars, select=c("wt","disp")))

x <- cbind(1,x) #Add column of 1 on the left (theta0)


thetas=(solve(t(x)%*%x))%*%t(x)%*%y #Applying Normal Equation

predictions=x%*%thetas


#Multiple variable linear regression using R's built-in functions 
model <- lm(mtcars$mpg ~ mtcars$wt + mtcars$disp) # Create linear model 


sum_squared_errors = sum((model$residuals)^2)

summary(model) #engine displacement has the lowest coefficient and significance value
