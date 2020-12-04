# hierNetGxE

The package is developed to fit a regularized regression model that we call **hierNetGxE** for the joint selection of gene-environment (G$\times$E) interactions based on the hierarchical lasso [Bien et al. (2013)]. The model focuses on a single environmental exposure and induces a "main-effect-before-interaction" hierarchical structure. Unlike the original hierarchical lasso model, which was designed for the gene-gene (G$\times$G) interaction case, the G$\times$E model has a simpler block-separable structure that  makes it possible to fit in large-scale applications. We developed and implemented an efficient fitting algorithm and screening rules that can discard large numbers of irrelevant predictors with high accuracy.  

## How to install:
```R
library(devtools)
devtools::install_github("NataliaZemlianskaia/hierNetGxE")
```