tol = 1e-4
grid_size = 10
max_iterations = 4000
alpha = 0.5

for (seed in 1:3) {
  for (family in c("gaussian", "binomial")) {
    data = data.gen(seed=seed, family=family)
    
    fit = gesso.fit(G=data$G_train,
                    E=data$E_train,
                    Y=data$Y_train,
                    tolerance=tol,
                    grid_size=grid_size,
                    alpha=alpha,
                    family=family,
                    max_iterations=max_iterations,
                    normalize=TRUE)
    
    expect_equal(sum(fit_data$has_converged != 1), 0)
    
    fit_normalized = gesso.fit(data_not_normalized$G_train,
                               data_not_normalized$E_train,
                               data_not_normalized$Y_train,
                               tolerance=tol,
                               grid_size=grid_size,
                               family=family,
                               max_iterations=max_iterations,
                               normalize=TRUE)
    expect_equal(sum(fit_normalized$has_converged != 1), 0)
    
    expect_lt(max(abs(fit_data_normalized$grid - fit_normalized$grid)), 1e-12)
    expect_lt(max(abs(fit_data_normalized$objective_value - fit_normalized$objective_value)), 1e-12)
    expect_lt(max(abs(fit_data_normalized$beta_0 - fit_normalized$beta_0)), 1e-12)
