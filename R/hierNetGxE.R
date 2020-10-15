hierNetGxE = function(G, E, Y, standardize=FALSE, grid=NULL, grid_size=10, grid_min_ratio=1e-4, family="gaussian",
                      tolerance=1e-4, max_iterations=10000, min_working_set_size=100) {
  if (is.null(grid)) {
    grid = 10^seq(-4, log10(0.1), length.out = grid_size)
  }
  return(fitModelRcpp(G, E, Y, standardize, grid, grid_size, grid_min_ratio, family, tolerance, max_iterations, min_working_set_size))
}
