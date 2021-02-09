library(readr)
library(caret)

heart <- read_csv("heart_failure_clinical_records_dataset.csv", 
                  col_types = cols(anaemia = col_factor(levels = c("0", "1")), 
                                   diabetes = col_factor(levels = c("0", "1")), 
                                   high_blood_pressure = col_factor(levels = c("0","1"))))

# 

# Data management of continous variables ----------------------------------------------------

# I) Log transformation for: creatinine_phosphokinase & serum_creatinine
heart$CP_log <- log(heart$creatinine_phosphokinase)
heart$SC_log <- log(heart$serum_creatinine)

# II) Scaling using population summary statistics
Cols <- names(heart[, c(1, 5, 7, 9, 14, 15)])
heart[Cols] <- lapply(heart[Cols], scale)

# III) Remove unused variables
heart <- heart[,-c(3, 8, 12)]

# Sampling ----------------------------------------------------------------------------------

# I) Unweighted sampling
set.seed(11)
trainIndex <- createDataPartition(heart$DEATH_EVENT, p = .7, 
                                  list = FALSE, 
                                  times = 1)
heart_train <- heart[trainIndex,]
heart_test  <- heart[-trainIndex,]

# Data management of categorical variables --------------------------------------------------

# I) Create dummy variables

# for training dataset
train_dm <- dummyVars(DEATH_EVENT ~ ., data = heart_train)
train_dm <- predict(train_dm, newdata = heart_train)

# for testing dataset
test_dm <- dummyVars(DEATH_EVENT ~ ., data = heart_test)
test_dm <- predict(test_dm, newdata = heart_test)




library(glmnet)

CV.Lasso <- cv.glmnet(x = train_dm,
                y = heart_train$DEATH_EVENT,
                family = "binomial",
                nfolds = 10,
                standardize = F,
                alpha = 1)

plot(CV.Lasso)
log(CV.Lasso$lambda.min)

Lasso <- glmnet(x = train_dm,
                y = heart_train$DEATH_EVENT,
                family = "binomial",
                standardize = F,
                alpha = 1,
                lambda = CV.Lasso$lambda.min)
Lasso$beta
Coef_Lasso <- predict(CV.Lasso, type = "coef", s = "lambda.min")


GLM <- glm(formula = DEATH_EVENT ~ .,
              data = heart_train,
              family = "binomial")
GLM$coefficients[-1]

plot(y = GLM$coefficients[-1], x = (1:11)+0.2, pch = "", axes = FALSE, xlim = c(0, 12), xlab = "Variable", ylab = "Coefficient")
rect(xleft = seq(0.5, 10.5, by = 1),
     xright = seq(1.5, 11.5, by = 1),
     ytop = -100,
     ybottom = 100,
     col = ggplot2::alpha(c("white", "bisque"), 0.3), border = F)
axis(2, c(0.5, 0, -0.5), col = NA, col.ticks = 1)
axis(1, (1:11)+0.1, col = NA, 
     labels = c("Age", "Anemia", "Diabetes", "Ejection", "HBP", "Platelets", "Sodium", "Sex", "Smoking", "Log(CP)", "Log(SC)"), 
     col.ticks = 1)
segments(x0 = 1:11, x1 = 1:11, y1 = Coef_Lasso[-c(1,3, 5, 8)], y0 = 0, lty = "dotted",)
segments(x0 = (1:11)+0.2, x1 = (1:11)+0.2, y1 = GLM$coefficients[-1], y0 = 0, lty = "dotted")
abline(h = 0)
points(y = GLM$coefficients[-1], x = (1:11)+0.2, col = "black", pch = 16, cex = 1.5)
points(y = Coef_Lasso[-c(1,3, 5, 8)], x = 1:11, col = "red", pch = c(16, 4, 4, 16, 4, 4, 16, 4, 4, 16), cex = 1.5, lwd = 2)

heart_train$DEATH_EVENT_pred <- predict(CV.Lasso, newx = train_dm, type = "class", s = "lambda.min")
heart_test$DEATH_EVENT_pred <- predict(CV.Lasso, newx = test_dm, type = "class", s = "lambda.min")

confusionMatrix(factor(heart_train$DEATH_EVENT), factor(heart_train$DEATH_EVENT_pred))
confusionMatrix(factor(heart_test$DEATH_EVENT), factor(heart_test$DEATH_EVENT_pred))

plot(heart_train$ejection_fraction)
