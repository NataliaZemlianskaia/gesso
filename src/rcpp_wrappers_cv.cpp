#include <Rcpp.h>
#include <RcppEigen.h>
#include <string.h>

#include "Solver.h"


// [[Rcpp::export]]
Rcpp::List fitModelCVRcpp(const Eigen::Map<Eigen::MatrixXd>& G,
                          const Eigen::Map<Eigen::VectorXd>& E,
                          const Eigen::Map<Eigen::VectorXd>& Y,
                          const Eigen::Map<Eigen::VectorXd>& weights,
                          const Eigen::Map<Eigen::VectorXi>& test_idx,
                          const Rcpp::LogicalVector& standardize,
                          const Eigen::VectorXd& grid,
                          const Rcpp::NumericVector& grid_size,
                          const Rcpp::NumericVector& grid_min_ratio,
                          const std::string& family,
                          double tolerance,
                          int max_iterations,
                          int min_working_set_size) {
  
  Solver solver(G, E, Y, weights);
  
  const int grid_size_squared = grid.size() * grid.size();
  
  Eigen::VectorXd test_loss(grid_size_squared);
  Eigen::VectorXd lambda_1(grid_size_squared);
  Eigen::VectorXd lambda_2(grid_size_squared);

  Eigen::VectorXd grid_lambda_1 = grid;
  std::sort(grid_lambda_1.data(), grid_lambda_1.data() + grid_lambda_1.size());
  std::reverse(grid_lambda_1.data(), grid_lambda_1.data() + grid_lambda_1.size());
  Eigen::VectorXd grid_lambda_2 = grid_lambda_1;
    
  int index = 0;
  int curr_solver_iterations;
  for (int i = 0; i < grid.size(); ++i) {
    for (int j = 0; j < grid.size(); ++j) {
      curr_solver_iterations = solver.solve(grid_lambda_1[i], grid_lambda_2[j], tolerance, max_iterations, min_working_set_size);
      
      lambda_1[index] = grid_lambda_1[i];
      lambda_2[index] = grid_lambda_2[j];
      test_loss[index] = solver.get_test_loss(test_idx) / test_idx.size();
      index++;
      
      if (index >= grid_size_squared) {
        break;
      }
    }
    std::reverse(grid_lambda_2.data(), grid_lambda_2.data() + grid_lambda_2.size());
    
    if (index >= grid_size_squared) {
      break;
    }
  }
  
  // collect results in list and return to R
  return Rcpp::List::create(
    Rcpp::Named("lambda_1") = lambda_1,
    Rcpp::Named("lambda_2") = lambda_2,
    Rcpp::Named("test_loss") = test_loss
  );
}