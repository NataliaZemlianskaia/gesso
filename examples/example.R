
library(hierNetGxE)
library(glmnet)

family = "gaussian"
#family = "binomial"
seed = 1
p = 1000
sample_size = 300
n_g_non_zero = 15
n_gxe_non_zero = 10

tolerance = 1e-4
grid_size = 20

## generate data: p main effects and p interaction effects 
## with n_g_non_zero non-zero main effects and n_gxe_non_zero non-zero interaction effects, 
## sample size equal to sample_size
data = data.gen(sample_size=sample_size, p=p, 
                n_g_non_zero=n_g_non_zero, n_gxe_non_zero=n_gxe_non_zero, 
                family=family, mode="strong_hierarchical",
                seed=seed,
                normalize=FALSE)

## tune the model over a 2D grid of hyper-parameters   
tune_model = hierNetGxE.cv(data$G_train, data$E_train, data$Y_train, 
                           family=family, grid_size=20, tolerance=tolerance,
                           parallel=TRUE, nfold=3,
                           normalize=TRUE,
                           seed=1)

## obtain interaction and main effect coefficietns corresponding to the best model  
model = hierNetGxE.coef(fit=tune_model$fit, lambda=tune_model$lambda_min)
gxe_coefficients = model$beta_gxe                      
g_coefficients = model$beta_g    

## check if all non-zero features were recovered by the model
cbind(data$Beta_GxE[data$Beta_GxE != 0], gxe_coefficients[data$Beta_GxE != 0])
cbind(data$Beta_G[data$Beta_G != 0], g_coefficients[data$Beta_G != 0])

## calculate principal selection metrics
selection = selection.metrics(data, g_coefficients, gxe_coefficients)

## compare with glmnet selection
set.seed(seed)
tune_model_glmnet = cv.glmnet(x=cbind(data$E_train, data$G_train, data$G_train * data$E_train),
                              y=data$Y_train,
                              nfolds=3,
                              family=family)

coef_glmnet = coef(tune_model_glmnet, tune_model_glmnet$lambda.min)
g_glmnet = coef_glmnet[3: (p + 2)]
gxe_glmnet = coef_glmnet[(p + 3): (2 * p + 2)]   

cbind(data$Beta_GxE[data$Beta_GxE != 0], gxe_glmnet[data$Beta_GxE != 0])
cbind(data$Beta_G[data$Beta_G != 0], g_glmnet[data$Beta_G != 0])

selection_glmnet = selection.metrics(data, g_glmnet, gxe_glmnet)

cbind(selection, selection_glmnet)
