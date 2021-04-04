tol = 1e-4
family = "binomial"
#family = "gaussian"
max_iterations = 10000
sample_size = 400
p = 10000

data = data.gen(seed=1, family=family, normalize=FALSE,
                sample_size=sample_size, p=p, n_g_non_zero=20, n_gxe_non_zero=20)

grid = 10^seq(-4, log10(1), length.out=10) 

start = Sys.time()
fit = gesso.fit(data$G_train, data$E_train, data$Y_train,
                     tolerance=tol, grid=grid, family=family, normalize=TRUE,
                     max_iterations=max_iterations)
stop_cd = Sys.time() - start; stop_cd
cat('Number not-converged: ', sum(fit$has_converged == 0))

start = Sys.time()
fit_alpha0.5 = gesso.fit(data$G_train, data$E_train, data$Y_train,
                tolerance=tol, grid=grid, alpha=0.5, family=family, normalize=TRUE,
                max_iterations=max_iterations)
stop_cd = Sys.time() - start; stop_cd
cat('Number not-converged for 1D grid: ', sum(fit_alpha0.5$has_converged == 0))

start = Sys.time()
cv_result = gesso.cv(data$G_train, data$E_train, data$Y_train,
                tolerance=tol, grid_size=20, family=family, normalize=TRUE,
                max_iterations=max_iterations)
stop_cd = Sys.time() - start; stop_cd
cat('Number not-converged: ', sum(cv_result$fit$has_converged == 0))

alpha = cv_result$lambda_min$lambda_2 / cv_result$lambda_min$lambda_1
alpha

start = Sys.time()
cv_result_alpha = gesso.cv(data$G_train, data$E_train, data$Y_train, alpha=alpha,
                     tolerance=tol, grid_size=20, family=family, normalize=TRUE,
                     max_iterations=max_iterations)
stop_cd = Sys.time() - start; stop_cd
cat('Number not-converged: ', sum(cv_result_alpha$fit$has_converged == 0))

coefficients = gesso.coefnum(cv_result, 50, less_than=FALSE)
coefficients_alpha = gesso.coefnum(cv_result_alpha, 50, less_than=FALSE)

cbind(selection.metrics(data$Beta_G, data$Beta_GxE, coefficients$beta_g, coefficients$beta_gxe),
      selection.metrics(data$Beta_G, data$Beta_GxE, 
                        coefficients_alpha$beta_g, coefficients_alpha$beta_gxe))


data = data.gen(seed=1, family=family)
start = Sys.time()
cv_result_auc = gesso.cv(data$G_train, data$E_train, data$Y_train,
                     tolerance=tol, grid_size=20, family=family, normalize=TRUE,
                     max_iterations=max_iterations, type_measure = 'auc')
stop_cd = Sys.time() - start; stop_cd
cat('Number not-converged: ', sum(cv_result$fit$has_converged == 0))

start = Sys.time()
cv_result_loss = gesso.cv(data$G_train, data$E_train, data$Y_train,
                         tolerance=tol, grid_size=20, family=family, normalize=TRUE,
                         max_iterations=max_iterations, type_measure = 'loss')
stop_cd = Sys.time() - start; stop_cd
cat('Number not-converged: ', sum(cv_result$fit$has_converged == 0))

coefficients_auc = gesso.coef(cv_result_auc$fit, cv_result_auc$lambda_min)
coefficients_loss = gesso.coef(cv_result_loss$fit, cv_result_loss$lambda_min)

cbind(selection.metrics(data$Beta_G, data$Beta_GxE, coefficients_auc$beta_g, coefficients_auc$beta_gxe),
      selection.metrics(data$Beta_G, data$Beta_GxE, 
                        coefficients_loss$beta_g, coefficients_loss$beta_gxe))

cv_result_auc$fit$objective_value[
  cv_result_auc$fit$lambda_2 == cv_result_auc$lambda_min$lambda_2 & 
    cv_result_auc$fit$lambda_1 == cv_result_auc$lambda_min$lambda_1]

cv_result_loss$fit$objective_value[
  cv_result_loss$fit$lambda_2 == cv_result_loss$lambda_min$lambda_2 & 
    cv_result_loss$fit$lambda_1 == cv_result_loss$lambda_min$lambda_1]

