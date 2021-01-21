#ifndef BINOMIAL_SOLVER_H
#define BINOMIAL_SOLVER_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <string>

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "Solver.h"
#include "GaussianSolver.h"
#include "SolverTypes.h"

namespace {
inline double xlogx(const double x) {
  return log(x) * x;
}

double sigmoid_scalar(const double z) {
  return 1.0 / (1.0 + exp(-z));
  //return abs(z) < 9 ? 1.0 / (1.0 + exp(-z)) : (z < 0 ? 0.0 : 1.0);
}

//template<typename T>
VecXd sigmoid(const VecXd& z) {
  return z.unaryExpr(std::ref(sigmoid_scalar));
}
}

template <typename TG>
class BinomialSolver : public Solver<TG> {
  
private:
  using Solver<TG>::n;
  using Solver<TG>::p;
  using Solver<TG>::G;
  using Solver<TG>::E;
  using Solver<TG>::Y;
  using Solver<TG>::weights_user;
  using Solver<TG>::normalize;
  
  using Solver<TG>::normalize_weights_g;
  using Solver<TG>::normalize_weights_e;
  
  using Solver<TG>::b_0;
  using Solver<TG>::b_e;
  using Solver<TG>::b_g;
  using Solver<TG>::b_gxe;
  using Solver<TG>::delta;
  
  using Solver<TG>::weights;
  using Solver<TG>::xbeta;
  using Solver<TG>::Z_w;
  
  using Solver<TG>::safe_set_g;
  using Solver<TG>::safe_set_gxe;
  using Solver<TG>::safe_set_zero;
  using Solver<TG>::working_set;
  
  using Solver<TG>::abs_nu_by_G_uptodate;
  
  using Solver<TG>::sum_w;
  using Solver<TG>::sum_E_w;
  using Solver<TG>::norm2_E_w;
  using Solver<TG>::denominator_E;  
  
  using Solver<TG>::norm2_G;
  using Solver<TG>::norm2_GxE;
  using Solver<TG>::G_by_GxE;
  using Solver<TG>::case1_A22_div_detA;
  using Solver<TG>::case1_A12_div_detA;  
  using Solver<TG>::case_3_A;
  using Solver<TG>::case_3_B;  
  using Solver<TG>::case_3_E;
  using Solver<TG>::case_3_F;    
  using Solver<TG>::case5_A22_div_detA;
  using Solver<TG>::case5_A12_div_detA;    
  
  using Solver<TG>::active_set;
  
  using Solver<TG>::temp_p;
  using Solver<TG>::temp_n;
  
  using Solver<TG>::update_intercept;
  using Solver<TG>::update_b_for_working_set;

protected:
  VecXd norm_G;
  VecXd norm_GxE;  

  double primal_objective;
  VecXd abs_nu_by_G;
  VecXd abs_nu_by_GxE;
  double x_opt;
  VecXd upperbound_nu_by_G;
  VecXd upperbound_nu_by_GxE;
  
  VecXd abs_inner_nu_by_G;
  VecXd abs_inner_nu_by_GxE;
  
  VecXd nu;
  
  public: BinomialSolver(const MapMat& G_,
                         const Eigen::Map<Eigen::VectorXd>& E_,
                         const Eigen::Map<Eigen::VectorXd>& Y_,
                         const Eigen::Map<Eigen::VectorXd>& weights_,
                         bool normalize_) :
    Solver<TG>(G_, E_, Y_, weights_, normalize_),
    norm_G(p),    
    norm_GxE(p),    
    abs_nu_by_G(p),
    abs_nu_by_GxE(p),
    upperbound_nu_by_G(p),
    upperbound_nu_by_GxE(p),
    abs_inner_nu_by_G(p),
    abs_inner_nu_by_GxE(p),
    nu(n) {
    
    init();
  }
    
    BinomialSolver(const MapSparseMat& G_,
                   const Eigen::Map<Eigen::VectorXd>& E_,
                   const Eigen::Map<Eigen::VectorXd>& Y_,
                   const Eigen::Map<Eigen::VectorXd>& weights_,
                   bool normalize_) :
    Solver<TG>(G_, E_, Y_, weights_, normalize_),
    norm_G(p),    
    norm_GxE(p),    
    abs_nu_by_G(p),
    abs_nu_by_GxE(p),
    upperbound_nu_by_G(p),
    upperbound_nu_by_GxE(p),
    abs_inner_nu_by_G(p),
    abs_inner_nu_by_GxE(p),
    nu(n) {
      init();
    }
    
    void init() {
      abs_nu_by_G_uptodate = false;
      if (normalize) {
        for (int i = 0; i < p; ++i) {
          normalize_weights_g[i] = 1.0 / std::sqrt(G.col(i).cwiseProduct(G.col(i)).dot(weights_user) - sqr(G.col(i).dot(weights_user)));
        }
        normalize_weights_e = 1.0 / std::sqrt(E.cwiseProduct(E).dot(weights_user) - sqr(E.dot(weights_user)));
      } else {
        normalize_weights_g.setOnes(p);
        normalize_weights_e = 1;
      }      
      
      for (int i = 0; i < G.cols(); ++i) {
        temp_n = G.col(i).cwiseProduct(G.col(i)) * sqr(normalize_weights_g[i]);
        norm2_G[i] = temp_n.dot(weights_user);
        temp_n = normalize_weights_e * temp_n.cwiseProduct(E);
        temp_n = normalize_weights_e * temp_n.cwiseProduct(E);
        norm2_GxE[i] = temp_n.dot(weights_user);
      }
      norm_G = norm2_G.cwiseSqrt();
      norm_GxE = norm2_GxE.cwiseSqrt();
    }
    
    virtual ~BinomialSolver() {}
    
    void update_weighted_variables() {
      sum_w = weights.sum();
      sum_E_w = normalize_weights_e * E.dot(weights);
      norm2_E_w = sqr(normalize_weights_e) * E.cwiseProduct(E).dot(weights);
      denominator_E = sum_w * norm2_E_w - sqr(sum_E_w);
      
      for (int i = 0; i < G.cols(); ++i) {
        temp_n = G.col(i).cwiseProduct(G.col(i)) * sqr(normalize_weights_g[i]);
        norm2_G[i] = temp_n.dot(weights);
        temp_n = normalize_weights_e * temp_n.cwiseProduct(E);
        G_by_GxE[i] = temp_n.dot(weights);
        temp_n = normalize_weights_e * temp_n.cwiseProduct(E);
        norm2_GxE[i] = temp_n.dot(weights);
      }
      // const VecXd case1_detA
      temp_p = norm2_G.cwiseProduct(norm2_GxE) - G_by_GxE.cwiseProduct(G_by_GxE);
      case1_A22_div_detA = norm2_GxE.cwiseQuotient(temp_p);
      case1_A12_div_detA = G_by_GxE.cwiseQuotient(temp_p);
      case_3_A = (norm2_G + norm2_GxE);
      case_3_B = 2 * G_by_GxE;
      // const VecXd case5_detA
      temp_p = (norm2_G.cwiseProduct(norm2_GxE) - G_by_GxE.cwiseProduct(G_by_GxE));
      case5_A22_div_detA = norm2_GxE.cwiseQuotient(temp_p);
      case5_A12_div_detA = G_by_GxE.cwiseQuotient(temp_p); 
    }
    
    int solve(double lambda_1, double lambda_2, double tolerance, int max_iterations, int min_working_set_size) {
      safe_set_g.setOnes(p);
      safe_set_gxe.setOnes(p);
      safe_set_zero.setOnes(p);
      
      int num_passes = 0;
      int working_set_size = 0;
      for (int i = 0; i < p; ++i) {
        working_set_size += int(b_g[i] != 0 || b_gxe[i] != 0);
      }
      if (working_set_size == 0) {
        working_set_size = min_working_set_size;
      }
      working_set.resize(0);
      
      double duality_gap, inner_duality_gap, max_diff_tolerance, max_diff;
      while (num_passes < max_iterations) {
        duality_gap = check_duality_gap(lambda_1, lambda_2, false);
        if (duality_gap < tolerance) {
          break;
        }

        update_working_set(lambda_1, lambda_2, duality_gap, working_set_size);
        working_set_size = std::min(2 * working_set_size, p);
        
        active_set.setZero(p);
        //max_diff_tolerance = tolerance * 10;
        max_diff_tolerance = tolerance;
        while (num_passes < max_iterations) {
          inner_duality_gap = check_duality_gap(lambda_1, lambda_2, true);
          if (inner_duality_gap < tolerance) {
            break;
          } else {
            update_quadratic_approximation();
            update_weighted_variables();
            
            case_3_E = G_by_GxE * (lambda_1 - lambda_2);
            case_3_F = (lambda_1 * norm2_GxE - lambda_2 * norm2_G);            
          }
          
          while (num_passes < max_iterations) {
            max_diff = update_b_for_working_set(lambda_1, lambda_2, false);
            num_passes += 1;

            if (max_diff < max_diff_tolerance) {
              break;
            }
            while (num_passes < max_iterations && max_diff >= max_diff_tolerance) {
              max_diff = update_b_for_working_set(lambda_1, lambda_2, true);
              num_passes += 1;
            }
          }
        }
      }
      return num_passes;
    }
    
    void update_quadratic_approximation() {
      temp_n = sigmoid(xbeta); // probabilities
      weights.array() = temp_n.array() * (1 - temp_n.array()) * weights_user.array();
      Z_w = xbeta.cwiseProduct(weights) + (Y - temp_n).cwiseProduct(weights_user);
    }
    
    double compute_dual_objective() {
      double result = 0;
      temp_n = Y - nu;
      for (int i = 0; i < n; ++i) {
        if (weights_user[i] != 0 && temp_n[i] > 0 && temp_n[i] < 1) {
          result -= weights_user[i] * (xlogx(temp_n[i]) + xlogx(1 - temp_n[i]));
        }
      }
      return result;
    }
    
    double naive_projection(double lambda_1, double lambda_2, const Eigen::Ref<VecXd>& abs_nu_by_G, const Eigen::Ref<VecXd>& abs_nu_by_GxE) {
      temp_p = (lambda_1 * abs_nu_by_GxE - lambda_2 * abs_nu_by_G).cwiseQuotient(abs_nu_by_GxE + abs_nu_by_G).cwiseMax(0).cwiseMin(lambda_1);
      double M = std::numeric_limits<double>::infinity();
      for (int i = 0; i < abs_nu_by_G.size(); ++i) {
        if (abs_nu_by_G[i] > 0) {
          M = std::min(M, (lambda_1 - temp_p[i]) / abs_nu_by_G[i]);
        }
        if (abs_nu_by_GxE[i] > 0) {
          M = std::min(M, (lambda_2 + temp_p[i]) / abs_nu_by_GxE[i]);
        }
      }
      double x_hat = find_scalar_for_naive_projection();
      double x_opt;
      if (std::abs(x_hat) <= M) {
        x_opt = x_hat;
      } else {
        x_opt = sign(x_hat) * M;
      }
      /*nu *= x_opt;
      double xxx_1 = compute_dual_objective();
      nu /= x_opt;
      std::cout << "custom x_hat=" << x_hat << ", x_opt= " << x_opt << " => " << xxx_1 << " | ";
      
      if (std::abs(1) <= M) {
        x_opt = 1;
      } else {
        x_opt = sign(1) * M;
      }   
      nu *= x_opt;
      double xxx_2 = compute_dual_objective();
      nu /= x_opt;
      std::cout << "custom x_hat=1, x_opt= " << x_opt << " => " << xxx_2 << "\n";*/
      
      return x_opt;
    }
    
    double find_scalar_for_naive_projection() {
      //return 1;
      return triple_dot_product(nu, Y.array() - 0.5, weights_user) / triple_dot_product(nu, nu, weights_user);
    }
    
    void update_nu(double lambda_1, double lambda_2) {
      //if (!abs_nu_by_G_uptodate) {
      abs_nu_by_G = (nu.cwiseProduct(weights_user).transpose() * G).cwiseAbs().transpose().cwiseProduct(normalize_weights_g);
      abs_nu_by_GxE = (nu.cwiseProduct(weights_user).cwiseProduct(E).transpose() * G).cwiseAbs().transpose().cwiseProduct(normalize_weights_g) * normalize_weights_e;
      //}
      x_opt = naive_projection(lambda_1, lambda_2, abs_nu_by_G, abs_nu_by_GxE);
      nu *= x_opt;
    }
    
    void update_inner_nu(double lambda_1, double lambda_2) {
      //    abs_inner_nu_by_G = (inner_nu_res * G(Eigen::all, working_set)).cwiseAbs();
      //    abs_inner_nu_by_GxE = (inner_nu_res.cwiseProduct(E) * G(Eigen::all, working_set)).cwiseAbs();
      // For details on resize vs conservativeResize see
      // https://stackoverflow.com/questions/34449805/reserve-dense-eigen-matrix
      //abs_inner_nu_by_G.conservativeResize(working_set.size());
      //abs_inner_nu_by_GxE.conservativeResize(working_set.size());
      abs_inner_nu_by_G.setZero(working_set.size());
      abs_inner_nu_by_GxE.setZero(working_set.size());
      //VecXd res_w
      temp_n = nu.cwiseProduct(weights_user);
      for (int i = 0; i < working_set.size(); ++i) {
        abs_inner_nu_by_G[i] = std::abs(G.col(working_set[i]).dot(temp_n)) * normalize_weights_g[working_set[i]];
        abs_inner_nu_by_GxE[i] = std::abs(G.col(working_set[i]).dot(temp_n.cwiseProduct(E))) * normalize_weights_g[working_set[i]] * normalize_weights_e;
      }
      double x_opt = naive_projection(lambda_1, lambda_2, abs_inner_nu_by_G, abs_inner_nu_by_GxE);
      nu *= x_opt;
    }
    
    double check_duality_gap(double lambda_1, double lambda_2, bool use_working_set) {
      nu = Y - sigmoid(xbeta);
      update_nu_for_intercept();
      double dual_objective;
      if (use_working_set) {
        update_inner_nu(lambda_1, lambda_2);  
      } else {
        update_nu(lambda_1, lambda_2);  
      }
      dual_objective = compute_dual_objective();
      primal_objective = (-Y.cwiseProduct(xbeta) + (xbeta.array().exp().log1p()).matrix()).dot(weights_user) + lambda_1 * (b_g.cwiseAbs().cwiseMax(b_gxe.cwiseAbs())).sum() + lambda_2 * b_gxe.cwiseAbs().sum();
      return primal_objective - dual_objective;
    }
    
    void update_nu_for_intercept() {
      double sum_w = weights_user.sum();
      double sum_E_w = normalize_weights_e * E.dot(weights_user);
      double norm2_E_w = sqr(normalize_weights_e) * E.cwiseProduct(E).dot(weights_user);
      double denominator_E = sum_w * norm2_E_w - sqr(sum_E_w);
      double sum_nu_w = nu.dot(weights_user);
      
      double b = (sum_w * triple_dot_product(E * normalize_weights_e, nu, weights_user) - 
        sum_E_w * sum_nu_w) / denominator_E;
      double a = (sum_nu_w - sum_E_w * b) / sum_w;
      
      nu = nu.array() - a;
      nu -= E * b * normalize_weights_e;
    }
    
    void update_working_set(double lambda_1, double lambda_2, double dual_gap, int working_set_size) {
      double r = std::sqrt(dual_gap / 2);
      //ArrayXd d_j = ((lambda_1 - abs_nu_by_G.array()) + (r * norm_GxE).array().max(lambda_2 - abs_nu_by_GxE.array())) / (norm_GxE + norm_G).array();
      ArrayXd d_j = (lambda_1 - lambda_2 - (x_opt * abs_nu_by_G).array() - (lambda_2 - r * norm_GxE.array()).max((x_opt * abs_nu_by_GxE).array())) / (norm_GxE + norm_G).array();
      
      upperbound_nu_by_G = x_opt * abs_nu_by_G + r * norm_G;
      upperbound_nu_by_GxE = x_opt * abs_nu_by_GxE + r * norm_GxE;
      safe_set_zero = (upperbound_nu_by_GxE.array() - lambda_2).max(0) < (lambda_1 - upperbound_nu_by_G.array());
      for (int i = 0; i < p; ++i) {
        safe_set_gxe[i] = safe_set_gxe[i] && (!safe_set_zero[i]) && (upperbound_nu_by_GxE[i] >= lambda_2);
        safe_set_g[i] = safe_set_g[i] && (!safe_set_zero[i]) && (upperbound_nu_by_G[i] >= lambda_1 || safe_set_gxe[i]);      
      }
      
      for (int i = 0; i < p; ++i) {
        if (b_gxe[i] != 0 && (!safe_set_gxe[i])) {
          xbeta -= normalize_weights_e * normalize_weights_g[i] * G.col(i).cwiseProduct(E) * b_gxe[i];
          abs_nu_by_G_uptodate = false;
          b_gxe[i] = 0;
        }
        if (b_g[i] != 0 && (!safe_set_g[i])) {
          xbeta -= normalize_weights_g[i] * G.col(i) * b_g[i];
          abs_nu_by_G_uptodate = false;
          b_g[i] = 0;
        }      
      }
      std::vector<int> working_set_tmp = argsort<ArrayXd>(d_j);
      int index;
      working_set.resize(0);
      for (int i = 0; i < p; ++i) {
        if (b_g[i] != 0 || b_gxe[i] != 0) {
          working_set.push_back(i);
        }
      }
      for (int i = 0; i < p; ++i) {
        index = working_set_tmp[i];
        if (b_g[index] == 0 && b_gxe[index] == 0 && safe_set_g[index] && working_set.size() < working_set_size) {
          working_set.push_back(index);  
        }
      }
    }
    
    virtual double get_value() {
      return primal_objective;
    }
    
    virtual double get_test_loss(const std::vector<int>& test_idx) {
      double test_loss = 0;
      int index;
      for (int i = 0; i < test_idx.size(); ++i) {
        index = test_idx[i];
        test_loss += -Y[index] * xbeta[index] + std::log1p(std::exp(xbeta[index]));
      }
      return test_loss;
    }
};

#endif // BINOMIAL_SOLVER_H