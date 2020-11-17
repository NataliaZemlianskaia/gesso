#include <Rcpp.h>
#include <RcppEigen.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "Solver.h"

// [[Rcpp::depends(RcppThread)]]
#include <RcppThread.h>

// [[Rcpp::export]]
void fitModelCVRcppSingleFold(const Eigen::Map<Eigen::MatrixXd>& G,
                                const Eigen::Map<Eigen::VectorXd>& E,
                                const Eigen::Map<Eigen::VectorXd>& Y,
                                const std::vector<int>& fold_ids,
                                const Rcpp::LogicalVector& standardize,
                                const Eigen::VectorXd& grid,
                                const std::string& family,
                                double tolerance,
                                int max_iterations,
                                int min_working_set_size,
                                int test_fold_id,
                                Eigen::MatrixXd& test_loss) {
  const int n = fold_ids.size();
  Eigen::VectorXd weights(n);
  std::vector<int> test_idx;
  for (int i = 0; i < n; ++i) {
    if (fold_ids[i] == test_fold_id) {
      weights[i] = 0;
      test_idx.push_back(i);
    } else {
      weights[i] = 1;
    }
  }
  weights = weights / weights.sum();
  Eigen::Map<Eigen::VectorXd> weights_map(weights.data(), n);
  
  Solver solver(G, E, Y, weights_map);
  
  const int grid_size_squared = grid.size() * grid.size();
  
  Eigen::VectorXd grid_lambda_1 = grid;
  std::sort(grid_lambda_1.data(), grid_lambda_1.data() + grid_lambda_1.size());
  std::reverse(grid_lambda_1.data(), grid_lambda_1.data() + grid_lambda_1.size());
  Eigen::VectorXd grid_lambda_2 = grid_lambda_1;
  
  int index = 0;
  int curr_solver_iterations;
  for (int i = 0; i < grid.size(); ++i) {
    for (int j = 0; j < grid.size(); ++j) {
      curr_solver_iterations = solver.solve(grid_lambda_1[i], grid_lambda_2[j], tolerance, max_iterations, min_working_set_size);
      
      test_loss(test_fold_id, index) = solver.get_test_loss(test_idx) / test_idx.size();
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
}

// [[Rcpp::export]]
Eigen::MatrixXd fitModelCVRcpp(const Eigen::Map<Eigen::MatrixXd>& G,
                                     const Eigen::Map<Eigen::VectorXd>& E,
                                     const Eigen::Map<Eigen::VectorXd>& Y,
                                     const Rcpp::LogicalVector& standardize,
                                     const Eigen::VectorXd& grid,
                                     const Rcpp::NumericVector& grid_size,
                                     const Rcpp::NumericVector& grid_min_ratio,
                                     const std::string& family,
                                     double tolerance,
                                     int max_iterations,
                                     int min_working_set_size,
                                     int nfolds,
                                     int seed,
                                     int ncores) {
  
  const int grid_size_squared = grid.size() * grid.size();
  Eigen::MatrixXd test_loss(nfolds, grid_size_squared);
  
  std::vector<int> fold_ids;
  for (int i = 0; i < G.rows(); ++i) {
    fold_ids.push_back(i % nfolds);
  }
  srand(seed);
  std::random_shuffle(fold_ids.begin(), fold_ids.end());
  
  if (ncores == 1) {
    for (int test_fold_id = 0; test_fold_id < nfolds; ++test_fold_id)
      fitModelCVRcppSingleFold(G, E, Y, fold_ids, standardize, grid, family,
                           tolerance, max_iterations,
                           min_working_set_size, test_fold_id, test_loss);
  } else {
    RcppThread::ThreadPool pool(ncores);
    for (int test_fold_id = 0; test_fold_id < nfolds; ++test_fold_id)
      pool.push([&, test_fold_id] {
        fitModelCVRcppSingleFold(G, E, Y, fold_ids, standardize, grid,
                             family, tolerance, max_iterations,
                             min_working_set_size, test_fold_id, test_loss);
      });
    pool.join();
  }
  return test_loss;
}
