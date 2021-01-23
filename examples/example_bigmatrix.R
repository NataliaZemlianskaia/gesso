library(bigmemory)
library(hierNetGxE)

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
grid = compute.grid(data$G_train, data$E_train, data$Y_train,
                    normalize=TRUE, grid_size=grid_size, grid_min_ratio=1e-4)

file.remove("g_train.bin")
file.remove("g_train.desc")
G_train = as.big.matrix(data$G_train,
                        backingfile="g_train.bin",
                        descriptorfile="g_train.desc")
is.filebacked(G_train)

# Fit
start = Sys.time()
fit = hierNetGxE.fit(data$G_train, data$E_train, data$Y_train,
                     tolerance=tolerance, grid=grid, family=family)
stop_cd = Sys.time() - start; stop_cd


start = Sys.time()
fit_bm = hierNetGxE.fit(G_train, data$E_train, data$Y_train,
                     tolerance=tolerance, grid=grid, family=family)
stop_cd = Sys.time() - start; stop_cd

summary(fit$objective_value - fit_bm$objective_value)

# CV
start = Sys.time()
tune_model = hierNetGxE.cv(data$G_train, data$E_train, data$Y_train, 
                           family=family, grid=grid, tolerance=tolerance,
                           parallel=TRUE, nfold=3,
                           normalize=TRUE, seed=1)
stop_cd = Sys.time() - start; stop_cd

start = Sys.time()
tune_model_bm = hierNetGxE.cv(G_train, data$E_train, data$Y_train, 
                              family=family, grid=grid, tolerance=tolerance,
                              parallel=TRUE, nfold=3,
                              normalize=TRUE, seed=1)
stop_cd = Sys.time() - start; stop_cd

summary(tune_model$cv_result$mean_loss - tune_model_bm$cv_result$mean_loss)
summary(tune_model$fit$objective_value - tune_model_bm2$fit$objective_value)

rm(G_train)
gc()
file.remove("g_train.bin")
file.remove("g_train.desc")
