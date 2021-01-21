# hierNetGxE

The package is developed to fit a regularized regression model that we call **hierNetGxE** for the joint selection of gene-environment (GxE) interactions based on the hierarchical lasso [Bien et al. (2013)]. The model focuses on a single environmental exposure and induces a "main-effect-before-interaction" hierarchical structure. Unlike the original hierarchical lasso model, which was designed for the gene-gene (GxG) interaction case, the GxE model has a simpler block-separable structure that  makes it possible to fit in large-scale applications. We developed and implemented an efficient fitting algorithm and screening rules that can discard large numbers of irrelevant predictors with high accuracy.


![](man/figures/hierNet_model_.png)

**hierNetGxE** model induces hierarchical selection of the (GxE) interaction terms via convex constraints added to the objective function. The model has two tuning parameters λ1 and λ2 responsible for the model sparsity with respect to main effects and interactions respectively.

## Installation
```R
## install.packages("devtools")

library(devtools)
devtools::install_github("NataliaZemlianskaia/hierNetGxE")
```
## Example
```R
library(hierNetGxE)

## generate the data: 1,000 main effects and 1,000 interaction effects 
## with 15 non-zero main effects and 10 non-zero interaction effects, sample size equal to 200
data = data.gen(sample_size=200, p=1000, 
                n_g_non_zero=15, n_gxe_non_zero=10, 
                family="gaussian", mode="strong_hierarchical")

## tune the model over a 2D grid of hyperparameters   
tune_model = hierNetGxE.cv(data$G_train, data$E_train, data$Y_train, 
                           grid_size=20, tolerance=1e-4,
                           parallel=TRUE, nfold=3,
                           normalize=TRUE,
                           seed=1)

## obtain interaction and main effect coefficietns corresponding to the best model  
coefficients = hierNetGxE.coef(fit=tune_model$fit, lambda=tune_model$lambda_min)
gxe_coefficients = coefficients$beta_gxe                      
g_coefficients = coefficients$beta_g    

## check if all non-zero features were recovered by the model
cbind(data$Beta_GxE[data$Beta_GxE != 0], gxe_coefficients[data$Beta_GxE != 0])

## calculate principal selection metrics
selection = selection.metrics(true_b_g=data$Beta_G, true_b_gxe=data$Beta_GxE, 
                              estimated_b_g=g_coefficients, estimated_b_gxe=gxe_coefficients)
cbind(selection)

##                selection
## b_g_non_zero    109      
## b_gxe_non_zero  43       
## mse_beta        0.3624545
## mse_beta_GxE    0.884622 
## sensitivity_g   1        
## specificity_g   0.9045685
## sensitivity_gxe 0.9      
## specificity_gxe 0.9656566
## precision_g     0.1376147
## precision_gxe   0.2093023
## auc_gxe         0.9232314
## auc_g           0.9999993
```
