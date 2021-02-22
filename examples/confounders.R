grid_size = 20
grid = 10^seq(-5, log10(1), length.out = grid_size)
p = 10
sample_size = 500 #, 200, 100)500,
family = "gaussian"
n_g_non_zero = 10
n_gxe_non_zero = 5
normalize = TRUE
n_steps = 20
#mode = "anti_hierarchical"
mode = "strong_hierarchical"
#mode = "hierarchical"
num_cofounders = 2
seed = 1

set.seed(seed)
dataset = data.gen(seed=i, sample_size=sample_size, p=p,
                   n_g_non_zero=n_g_non_zero, n_gxe_non_zero=n_gxe_non_zero,
                   family=family, 
                   normalize=normalize, mode=mode,
                   n_confounders=2)

start = Sys.time()
hiernet_fit = hierNetGxE.fit(G=dataset$G_train, E=dataset$E_train, Y=dataset$Y_train,
                             C=dataset$C_train,
                             grid_size=grid_size, tol=1e-4, family=family)
stop = Sys.time() - start; cat("hierNet: "); print(stop)



start = Sys.time()
hiernet_fit = hierNetGxE.fit(G=dataset$G_train, E=rep(0, sample_size), Y=dataset$Y_train,
                             C=dataset$C_train,
                             grid=grid, tol=1e-4, family=family, normalize=FALSE)
stop = Sys.time() - start; cat("hierNet: "); print(stop)



set.seed(seed)
dataset = data.gen(seed=i, sample_size=sample_size, p=p,
                   n_g_non_zero=n_g_non_zero, n_gxe_non_zero=n_gxe_non_zero,
                   family=family, 
                   normalize=TRUE, mode=mode,
                   n_confounders=0)

start = Sys.time()
hiernet_fit = hierNetGxE.fit(G=dataset$G_train, E=rep(0, sample_size), Y=dataset$Y_train,
                             C=dataset$C_train,
                             grid=grid, tol=1e-4, family=family, normalize=FALSE)
stop = Sys.time() - start; cat("hierNet: "); print(stop)


test = glmnet(x=cbind(dataset$G_train), y=dataset$Y_train, lambda=grid, thresh=1e-14)
