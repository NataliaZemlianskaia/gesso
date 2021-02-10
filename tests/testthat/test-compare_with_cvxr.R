context("compare with cvxr")

test_that("training loss is similar to the fit by CVXR", {
  grid = 10^seq(-4, log10(1), length.out=10) 
  tol = 1e-6
  max_iterations = 10000

  for (family in c("gaussian", "binomial")){
    for (seed in 1:3) {
      file_name = paste0("testdata/", seed, "_", family, "_data.rds")
      data = readRDS(file_name)

      fit = hierNetGxE.fit(data$G_train, data$E_train, data$Y_train,
                           tolerance=tol, grid=grid, family=family, 
                           normalize=FALSE)
      
      file_name = paste0("testdata/", seed, "_", family, "_cvxr_results.rds")
      cvxr_fit = readRDS(file_name)
      
      expect_lt(max(fit$objective_value - cvxr_fit), tol)
    }
  }
})
