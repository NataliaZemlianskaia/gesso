
library(hierNetGxE)
library(glmnet)

#family = "gaussian"
family = "binomial"
seed = 1
p = 100000
sample_size = 1000
n_g_non_zero = 30
n_gxe_non_zero = 20
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

## OR obtain interaction and main effect coefficietns corresponding to the target model  
coefficients = hierNetGxE.coefnum(tune_model, 100, less_than=FALSE)
gxe_coefficients = coefficients$beta_gxe                      
g_coefficients = coefficients$beta_g    

## calculate principal selection metrics
selection = selection.metrics(data$Beta_G, data$Beta_GxE, g_coefficients, gxe_coefficients)
cbind(selection)

## compare with glmnet selection
set.seed(seed)
start = Sys.time()
tune_model_glmnet = cv.glmnet(x=cbind(data$E_train, data$G_train, data$G_train * data$E_train),
                              y=data$Y_train,
                              nfolds=3,
                              family=family)
Sys.time() - start

coef_glmnet = coef(tune_model_glmnet, tune_model_glmnet$lambda.min)
g_glmnet = coef_glmnet[3: (p + 2)]
gxe_glmnet = coef_glmnet[(p + 3): (2 * p + 2)]   
selection_glmnet = selection.metrics(data$Beta_G, data$Beta_GxE, g_glmnet, gxe_glmnet)

## check if all non-zero features were recovered by the models
cbind(data$Beta_GxE[data$Beta_GxE != 0], gxe_coefficients[data$Beta_GxE != 0], 
      gxe_glmnet[data$Beta_GxE != 0])
cbind(data$Beta_G[data$Beta_G != 0], g_coefficients[data$Beta_G != 0],
      g_glmnet[data$Beta_G != 0])

(data$Beta_GxE[order(abs(gxe_glmnet), decreasing=TRUE)])[1:n_gxe_non_zero]
(data$Beta_GxE[order(abs(gxe_coefficients), decreasing=TRUE)])[1:n_gxe_non_zero]

## compare selection metrics
cbind(selection, selection_glmnet)

## Prediction
coefficients = hierNetGxE.coef(tune_model$fit, tune_model$lambda_min)
beta_0 = coefficients$beta_0; beta_e = coefficients$beta_e                   
beta_g = coefficients$beta_g; beta_gxe = coefficients$beta_gxe     

new_G = data$G_test; new_E = data$E_test
new_Y = hierNetGxE.predict(beta_0, beta_e, beta_g, beta_gxe, new_G, new_E, family=family)

## for "gaussian" family
test_R2_hierNetGxE = cor(new_Y, data$Y_test)^2
## for "binomial" family
accuracy_hierNetGxE = sum(new_Y > 0.5 & data$Y_test == 1) / length(data$Y_test)

new_Y_glmnet = predict(tune_model_glmnet, newx=cbind(new_E, new_G, new_G * new_E), 
                       s=tune_model_glmnet$lambda.min)
test_R2_glmnet = cor(new_Y_glmnet[,1], data$Y_test)^2
accuracy_glmnet = sum(new_Y_glmnet > 0.5 & data$Y_test == 1) / length(data$Y_test)

cbind(test_R2_hierNetGxE, test_R2_glmnet)
cbind(accuracy_hierNetGxE, accuracy_glmnet)



