#include <bigmemory/MatrixAccessor.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

#include <RcppThread.h>

#include "SolverTypes.h"
#include "Solver.h"
#include "GaussianSolver.h"
#include "BinomialSolver.h"

// [[Rcpp::depends(RcppEigen, RcppThread, BH, bigmemory)]]

template <typename TG>
void fitModelCVRcppSingleFold(const TG& G,
                              const Eigen::Map<Eigen::VectorXd>& E,
                                const Eigen::Map<Eigen::VectorXd>& Y,
                                const std::vector<int>& fold_ids,
                                const Rcpp::LogicalVector& normalize,
                                const Eigen::VectorXd& grid,
                                const std::string& family,
                                double tolerance,
                                int max_iterations,
                                int min_working_set_size,
                                int test_fold_id,
                                Eigen::MatrixXd& test_loss,
                                Eigen::MatrixXi& beta_g_nonzero,
                                Eigen::MatrixXi& beta_gxe_nonzero) {
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
  
  std::unique_ptr<Solver<TG> > solver;
  if (family == "gaussian") {
    solver.reset(
      new GaussianSolver<TG>(G, E, Y, weights_map, normalize[0]));
  } 
  else if (family == "binomial") {
    solver.reset(
      new BinomialSolver<TG>(G, E, Y, weights_map, normalize[0]));
  }

  const int grid_size_squared = grid.size() * grid.size();
  
  Eigen::VectorXd grid_lambda_1 = grid;
  std::sort(grid_lambda_1.data(), grid_lambda_1.data() + grid_lambda_1.size());
  std::reverse(grid_lambda_1.data(), grid_lambda_1.data() + grid_lambda_1.size());
  Eigen::VectorXd grid_lambda_2 = grid_lambda_1;
  
  int index = 0;
  int curr_solver_iterations;
  for (int i = 0; i < grid.size(); ++i) {
    for (int j = 0; j < grid.size(); ++j) {
      curr_solver_iterations = solver->solve(grid_lambda_1[i], grid_lambda_2[j], tolerance, max_iterations, min_working_set_size);
      
      test_loss(test_fold_id, index) = solver->get_test_loss(test_idx) / test_idx.size();
      beta_g_nonzero(test_fold_id, index) = solver->get_b_g_non_zero();
      beta_gxe_nonzero(test_fold_id, index) = solver->get_b_gxe_non_zero();
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

template <typename TG>
Rcpp::List fitModelCVRcpp(const TG& G,
                          const Eigen::Map<Eigen::VectorXd>& E,
                          const Eigen::Map<Eigen::VectorXd>& Y,
                          const Rcpp::LogicalVector& normalize,
                          const Eigen::VectorXd& grid,
                          const std::string& family,
                          double tolerance,
                          int max_iterations,
                          int min_working_set_size,
                          int nfolds,
                          int seed,
                          int ncores) {
  const int grid_size_squared = grid.size() * grid.size();
  Eigen::MatrixXd test_loss(nfolds, grid_size_squared);
  Eigen::MatrixXi beta_g_nonzero(nfolds, grid_size_squared);
  Eigen::MatrixXi beta_gxe_nonzero(nfolds, grid_size_squared);

  std::vector<int> fold_ids;
  for (int i = 0; i < G.rows(); ++i) {
    fold_ids.push_back(i % nfolds);
  }
  std::shuffle(fold_ids.begin(), fold_ids.end(), std::default_random_engine(seed));

  if (ncores == 1) {
    for (int test_fold_id = 0; test_fold_id < nfolds; ++test_fold_id)
      fitModelCVRcppSingleFold<TG>(G, E, Y, fold_ids, normalize, grid, family,
                           tolerance, max_iterations, min_working_set_size,
                           test_fold_id, test_loss, beta_g_nonzero, beta_gxe_nonzero);
  } else {
    RcppThread::ThreadPool pool(ncores);
    for (int test_fold_id = 0; test_fold_id < nfolds; ++test_fold_id)
      pool.push([&, test_fold_id] {
        fitModelCVRcppSingleFold<TG>(G, E, Y, fold_ids, normalize, grid,
                             family, tolerance, max_iterations, min_working_set_size,
                             test_fold_id, test_loss, beta_g_nonzero, beta_gxe_nonzero);
      });
    pool.join();
  }
  return Rcpp::List::create(
    Rcpp::Named("test_loss") = test_loss,
    Rcpp::Named("beta_g_nonzero") = beta_g_nonzero,
    Rcpp::Named("beta_gxe_nonzero") = beta_gxe_nonzero
  );  
}

// [[Rcpp::export]]
Rcpp::List fitModelCV(SEXP G,
                      const Eigen::Map<Eigen::VectorXd>& E,
                      const Eigen::Map<Eigen::VectorXd>& Y,
                      const Rcpp::LogicalVector& normalize,
                      const Eigen::VectorXd& grid,
                      const std::string& family,
                      double tolerance,
                      int max_iterations,
                      int min_working_set_size,
                      int nfolds,
                      int seed,
                      int ncores,
                      int mattype_g) {
  if (mattype_g == 1) {
    return fitModelCVRcpp<MapSparseMat>(Rcpp::as<MapSparseMat>(G), E, Y,
                                    normalize, grid,
                                    family, tolerance, max_iterations, min_working_set_size,
                                    nfolds, seed, ncores);    
  } if (mattype_g == 2) {
    Rcpp::S4 G_info(G);
    Rcpp::XPtr<BigMatrix> xptr((SEXP) G_info.slot("address"));
    MapMat Gmap((const double *)xptr->matrix(), xptr->nrow(), xptr->ncol());
    return fitModelCVRcpp<MapMat>(Gmap, E, Y, normalize, grid,
                                  family, tolerance, max_iterations, min_working_set_size,
                                  nfolds, seed, ncores);
  } else {
    Rcpp::NumericMatrix G_mat(G);
    MapMat Gmap((const double *) &G_mat[0], G_mat.rows(), G_mat.cols());
    return fitModelCVRcpp<MapMat>(Gmap, E, Y, normalize, grid,
                                  family, tolerance, max_iterations, min_working_set_size,
                                  nfolds, seed, ncores);
  }
}