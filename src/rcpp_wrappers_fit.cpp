#include <bigmemory/MatrixAccessor.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <string.h>

#include "SolverTypes.h"
#include "Solver.h"
#include "GaussianSolver.h"
#include "BinomialSolver.h"

// [[Rcpp::depends(RcppEigen, BH, bigmemory)]]

template <typename TG>
Rcpp::List fitModelRcpp(const TG& G,
                        const Eigen::Map<Eigen::VectorXd>& E,
                        const Eigen::Map<Eigen::VectorXd>& Y,
                        const Eigen::Map<Eigen::VectorXd>& weights,
                        const Rcpp::LogicalVector& normalize,
                        const Eigen::VectorXd& grid,
                        const std::string& family,
                        double tolerance,
                        int max_iterations,
                        int min_working_set_size) {
  
  std::unique_ptr<Solver<TG> > solver;
  if (family == "gaussian") {
    solver.reset(
      new GaussianSolver<TG>(G, E, Y, weights, normalize[0]));
  } 
  else if (family == "binomial") {
    solver.reset(
      new BinomialSolver<TG>(G, E, Y, weights, normalize[0]));
  }
  
  const int grid_size_squared = grid.size() * grid.size();

  Eigen::VectorXd beta_0(grid_size_squared);
  Eigen::VectorXd beta_e(grid_size_squared);
  Eigen::MatrixXd beta_g(grid_size_squared, G.cols());
  Eigen::MatrixXd beta_gxe(grid_size_squared, G.cols());
  Eigen::VectorXd lambda_1(grid_size_squared);
  Eigen::VectorXd lambda_2(grid_size_squared);
  Eigen::VectorXd working_set_size(grid_size_squared);
  Eigen::VectorXd num_iterations(grid_size_squared);
  Eigen::VectorXd num_fitered_by_safe_g(grid_size_squared);
  Eigen::VectorXd num_fitered_by_safe_gxe(grid_size_squared);
  Eigen::VectorXd objective_value(grid_size_squared);
  
  Eigen::VectorXd grid_lambda_1 = grid;
  std::sort(grid_lambda_1.data(), grid_lambda_1.data() + grid_lambda_1.size());
  std::reverse(grid_lambda_1.data(), grid_lambda_1.data() + grid_lambda_1.size());
  Eigen::VectorXd grid_lambda_2 = grid_lambda_1;
  
  int index = 0;
  int curr_solver_iterations;
  for (int i = 0; i < grid.size(); ++i) {
    for (int j = 0; j < grid.size(); ++j) {
      curr_solver_iterations = solver->solve(grid_lambda_1[i], grid_lambda_2[j], tolerance, max_iterations, min_working_set_size);

      beta_0[index] = solver->get_b_0();
      beta_e[index] = solver->get_b_e();
      beta_g.row(index) = solver->get_b_g();
      beta_gxe.row(index) = solver->get_b_gxe();
      lambda_1[index] = grid_lambda_1[i];
      lambda_2[index] = grid_lambda_2[j];
      num_iterations[index] = curr_solver_iterations;
      working_set_size[index] = solver->get_working_set_size();
      num_fitered_by_safe_g[index] = solver->get_num_fitered_by_safe_g();
      num_fitered_by_safe_gxe[index] = solver->get_num_fitered_by_safe_gxe();
      objective_value[index] = solver->get_value();
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
    Rcpp::Named("beta_0") = beta_0,
    Rcpp::Named("beta_e") = beta_e,
    Rcpp::Named("beta_g") = beta_g,
    Rcpp::Named("beta_gxe") = beta_gxe,
    Rcpp::Named("lambda_1") = lambda_1,
    Rcpp::Named("lambda_2") = lambda_2,
    Rcpp::Named("num_iterations") = num_iterations,
    Rcpp::Named("working_set_size") = working_set_size,
    Rcpp::Named("num_fitered_by_safe_g") = num_fitered_by_safe_g,
    Rcpp::Named("num_fitered_by_safe_gxe") = num_fitered_by_safe_gxe,
    Rcpp::Named("objective_value") = objective_value,    
    Rcpp::Named("grid") = grid
  );
}

// [[Rcpp::export]]
Rcpp::List fitModel(SEXP G,
                    const Eigen::Map<Eigen::VectorXd>& E,
                    const Eigen::Map<Eigen::VectorXd>& Y,
                    const Eigen::Map<Eigen::VectorXd>& weights,
                    const Rcpp::LogicalVector& normalize,
                    const Eigen::VectorXd& grid,
                    const std::string& family,
                    double tolerance,
                    int max_iterations,
                    int min_working_set_size,
                    int mattype_g) {
  if (mattype_g == 1) {
    return fitModelRcpp<MapSparseMat>(Rcpp::as<MapSparseMat>(G), E, Y,
                                            weights, normalize, grid,
                                            family, tolerance, max_iterations, min_working_set_size);    
  } if (mattype_g == 2) {
    Rcpp::S4 G_info(G);
    Rcpp::XPtr<BigMatrix> xptr((SEXP) G_info.slot("address"));
    MapMat Gmap((const double *)xptr->matrix(), xptr->nrow(), xptr->ncol());
    return fitModelRcpp<MapMat>(Gmap, E, Y, weights, normalize, grid,
                                family, tolerance, max_iterations, min_working_set_size);
  } else {
    Rcpp::NumericMatrix G_mat(G);
    MapMat Gmap((const double *) &G_mat[0], G_mat.rows(), G_mat.cols());
    return fitModelRcpp<MapMat>(Gmap, E, Y, weights, normalize, grid,
                                family, tolerance, max_iterations, min_working_set_size);
  }
}