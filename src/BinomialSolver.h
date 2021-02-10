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
  /*if (std::fabs(z) > 9) {
    return z < 0 ? 0 : 1;
  }*/
  // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
  if (z >= 0) {
    return 1 / (1 + std::exp(-z));
  } else {
    const double exp_z = std::exp(z);
    return exp_z / (1 + exp_z);
  }
}

VecXd sigmoid(const VecXd& z) {
  return z.unaryExpr(std::ref(sigmoid_scalar));
}
  
double log_one_plus_exp_scalar(const double z) {
  // http://sachinashanbhag.blogspot.com/2014/05/numerically-approximation-of-log-1-expy.html
  if (z > 35) {
    return z;
  } else if (z > -10) {
    return std::log1p(std::exp(z));
  } else {
    return std::exp(z);
  }
}
  
VecXd log_one_plus_exp(const VecXd& z) {
  return z.unaryExpr(std::ref(log_one_plus_exp_scalar));
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
  
  using Solver<TG>::norm_G;
  using Solver<TG>::norm_GxE;
  
  using Solver<TG>::active_set;
  
  using Solver<TG>::temp_p;
  using Solver<TG>::temp_n;
  
  using Solver<TG>::update_intercept;
  using Solver<TG>::update_b_for_working_set;
  using Solver<TG>::update_weighted_variables;

protected:
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
    abs_nu_by_G(p),
    abs_nu_by_GxE(p),
    upperbound_nu_by_G(p),
    upperbound_nu_by_GxE(p),
    abs_inner_nu_by_G(p),
    abs_inner_nu_by_GxE(p),
    nu(n) {}
    
    BinomialSolver(const MapSparseMat& G_,
                   const Eigen::Map<Eigen::VectorXd>& E_,
                   const Eigen::Map<Eigen::VectorXd>& Y_,
                   const Eigen::Map<Eigen::VectorXd>& weights_,
                   bool normalize_) :
    Solver<TG>(G_, E_, Y_, weights_, normalize_),
    abs_nu_by_G(p),
    abs_nu_by_GxE(p),
    upperbound_nu_by_G(p),
    upperbound_nu_by_GxE(p),
    abs_inner_nu_by_G(p),
    abs_inner_nu_by_GxE(p),
    nu(n) {}
    
    virtual ~BinomialSolver() {}
    
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
        max_diff_tolerance = tolerance;
        int num_updates_b_for_working_set = 0;
        bool is_first_iteration = true;
        while (num_passes < max_iterations) {
          inner_duality_gap = check_duality_gap(lambda_1, lambda_2, true);
          if (inner_duality_gap < tolerance) {
            break;
          } else {
            if (!is_first_iteration && num_updates_b_for_working_set <= 1) {
              max_diff_tolerance /= 4;
            } else {
              update_quadratic_approximation();
              update_weighted_variables();
            }
            num_updates_b_for_working_set = 0;
            is_first_iteration = false;
          }
          
          while (num_passes < max_iterations) {
            max_diff = update_b_for_working_set(lambda_1, lambda_2, false);
            num_passes += 1;
            ++num_updates_b_for_working_set;
            
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
      int mis_classifications = 0;
      for (int i = 0; i < n; ++i) {
        if (temp_n[i] >= 0.5 && Y[i] == 0) {
          mis_classifications += 1;
        }
        if (temp_n[i] < 0.5 && Y[i] == 1) {
          mis_classifications += 1;
        }        
      }
      /*std::cout << "- weights= ";
      for (int i = 0; i < n; ++i) {
        std::cout << weights[i] << ", ";
        if (weights[i] < 1e-7) {
          weights[i] = 1e-7;
        }
      }
      std::cout << "\n";*/
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
      primal_objective = (-Y.cwiseProduct(xbeta) + log_one_plus_exp(xbeta)).dot(weights_user) + lambda_1 * (b_g.cwiseAbs().cwiseMax(b_gxe.cwiseAbs())).sum() + lambda_2 * b_gxe.cwiseAbs().sum();
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
        // TODO: fix below
        test_loss += -Y[index] * xbeta[index] + std::log1p(std::exp(xbeta[index]));
      }
      return test_loss;
    }
};

#endif // BINOMIAL_SOLVER_H