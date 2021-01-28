
library(hierNetGxE)
library(glmnet)
library(ggplot2)


family = "gaussian"
sample_size = 150; p = 400; n_g_non_zero = 10; n_gxe_non_zero = 5

data = data.gen(seed=1, sample_size=sample_size, p=p, 
                n_g_non_zero=n_g_non_zero, n_gxe_non_zero=n_gxe_non_zero, 
                mode = "strong_hierarchical",
                family=family)

cbind(data$Beta_G[data$Beta_G!=0], data$Beta_GxE[data$Beta_G!=0])
cbind(data$Beta_G[data$Beta_GxE!=0], data$Beta_GxE[data$Beta_GxE!=0])
summary(data$Y_train)
  
################## selection ################## 

## tune the model over a 2D grid of hyper-parameters
tolerance = 1e-4
start = Sys.time()
tune_model = hierNetGxE.cv(data$G_train, data$E_train, data$Y_train, 
                           family=family, grid_size=20, tolerance=tolerance,
                           parallel=TRUE, nfold=3,
                           normalize=TRUE,
                           seed=1)
Sys.time() - start

## obtain interaction and main effect coefficietns corresponding to the best model  
coefficients = hierNetGxE.coef(fit=tune_model$fit, lambda=tune_model$lambda_min)
gxe_coefficients = coefficients$beta_gxe                      
g_coefficients = coefficients$beta_g   

## check if all non-zero features were recovered by the model
cbind(data$Beta_GxE[data$Beta_GxE != 0], gxe_coefficients[data$Beta_GxE != 0])
cbind(data$Beta_G[data$Beta_G != 0], g_coefficients[data$Beta_G != 0])

(data$Beta_GxE[order(abs(gxe_coefficients), decreasing=TRUE)])[1:20]

## calculate principal selection metrics
selection_hierNetGxE = selection.metrics(true_b_g=data$Beta_G, true_b_gxe=data$Beta_GxE, 
                              estimated_b_g=g_coefficients, estimated_b_gxe=gxe_coefficients)

cbind(selection_hierNetGxE)

## compare with the standart Lasso (glmnet) #alpha=1 is a default value 
set.seed(1)
tune_model_glmnet = cv.glmnet(x=cbind(data$E_train, data$G_train, data$G_train * data$E_train),
                              y=data$Y_train,
                              nfolds=3,
                              family=family)

coef_glmnet = coef(tune_model_glmnet, s=tune_model_glmnet$lambda.min)
g_glmnet = coef_glmnet[3: (p + 2)]
gxe_glmnet = coef_glmnet[(p + 3): (2 * p + 2)]   

cbind(data$Beta_GxE[data$Beta_GxE != 0], gxe_glmnet[data$Beta_GxE != 0])
cbind(data$Beta_G[data$Beta_G != 0], g_glmnet[data$Beta_G != 0])

selection_glmnet = selection.metrics(data$Beta_G, data$Beta_GxE, g_glmnet, gxe_glmnet)

cbind(selection_hierNetGxE, selection_glmnet)

(data$Beta_GxE[order(abs(gxe_coefficients), decreasing=TRUE)])[1:20]
(data$Beta_GxE[order(abs(gxe_glmnet), decreasing=TRUE)])[1:20]

## obtain interaction and main effect coefficietns corresponding to the target GxE model  
coefficients = hierNetGxE.coefnum(cv_model=tune_model, target_b_gxe_non_zero=10)
gxe_coefficients = coefficients$beta_gxe                      
g_coefficients = coefficients$beta_g   

## calculate principal selection metrics
selection_hierNetGxE = selection.metrics(true_b_g=data$Beta_G, true_b_gxe=data$Beta_GxE, 
                                         estimated_b_g=g_coefficients, estimated_b_gxe=gxe_coefficients)

cbind(selection_hierNetGxE, selection_glmnet)

################## prediction ################## 
coefficients = hierNetGxE.coef(tune_model$fit, tune_model$lambda_min)
beta_0 = coefficients$beta_0; beta_e = coefficients$beta_e                   
beta_g = coefficients$beta_g; beta_gxe = coefficients$beta_gxe     

new_G = data$G_test; new_E = data$E_test
new_Y = hierNetGxE.predict(beta_0, beta_e, beta_g, beta_gxe, new_G, new_E)
test_R2_hierNetGxE = cor(new_Y, data$Y_test)^2

## compare with the Lasso (glmnet)

new_Y_glmnet = predict(tune_model_glmnet, newx=cbind(new_E, new_G, new_G * new_E), 
                       s=tune_model_glmnet$lambda.min)
test_R2_glmnet = cor(new_Y_glmnet[,1], data$Y_test)^2

cbind(test_R2_hierNetGxE, test_R2_glmnet)



################## sparse matrix option ################## 
sample_size = 600; p = 30000
pG = 0.03
data = data.gen(seed=1, sample_size=sample_size, p=p, 
                n_g_non_zero=n_g_non_zero, n_gxe_non_zero=n_gxe_non_zero, 
                mode = "strong_hierarchical",
                pG=pG,
                family=family)
sum(colSums(data$G_train) == 0)
sum(data$G_train != 0) / (sample_size * p)

#G = data$G_train; E = data$E_train; Y = data$Y_train

start = Sys.time()
fit = hierNetGxE.fit(G=data$G_train, E=data$E_train, Y=data$Y_train, 
                     tolerance=tolerance,
                     normalize=TRUE)
time_non_sparse = Sys.time() - start; time_non_sparse


G_train_sparse = as(data$G_train, "dgCMatrix")

start = Sys.time()
fit = hierNetGxE.fit(G=G_train_sparse, E=data$E_train, Y=data$Y_train, 
                     tolerance=tolerance,
                     normalize=TRUE)
time_sparse = Sys.time() - start; time_sparse

(as.numeric(time_non_sparse) * 60) / as.numeric(time_sparse)

################## working set size ################## 

plot(fit$beta_g_nonzero, pch=19, cex=0.4, 
     ylab="num of non-zero features", xlab="lambdas path")
points(fit$beta_gxe_nonzero, pch=19, cex=0.4, col="red")
legend("topleft", legend=c("G features", "GxE features"), col=c("black", "red"), pch=19)

plot(fit$working_set_size, pch=19, cex=0.4, 
     ylab="num of non-zero features", xlab="lambdas path")

sum(fit$working_set_size > 3000)/ length(fit$working_set_size)

df = data.frame(lambda_1_factor = factor(fit$lambda_1),
                lambda_2_factor = factor(fit$lambda_2),
                ws = fit$working_set_size)

log_0 = function(x){
  return(ifelse(x == 0, 0, log10(x)))
}

ggplot(df, aes(lambda_1_factor, lambda_2_factor, fill=log_0(ws))) + 
  scale_fill_distiller(palette = "RdBu") +
  scale_x_discrete("lambda_1", breaks=c(1)) +
  scale_y_discrete("lambda_2", breaks=c(1)) +
  labs(fill='log WS') +
  geom_tile()


################## bigmatrix option ################## 

