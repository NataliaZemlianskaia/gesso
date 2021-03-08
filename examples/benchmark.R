tol = 1e-4
family = "binomial"
#family = "gaussian"
max_iterations = 10000
sample_size = 400
p = 40000

data = data.gen(seed=1, family=family, normalize=FALSE,
                sample_size=sample_size, p=p, n_g_non_zero=10, n_gxe_non_zero=5)

grid = 10^seq(-4, log10(1), length.out=10) 

start = Sys.time()
fit = gesso.fit(data$G_train, data$E_train, data$Y_train,
                     tolerance=tol, grid=grid, family=family, normalize=TRUE,
                     max_iterations=max_iterations)
stop_cd = Sys.time() - start; stop_cd
cat('Number not-converged: ', sum(fit$has_converged == 0))

