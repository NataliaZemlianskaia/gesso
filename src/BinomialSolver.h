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
  using Solver<TG>::normalize;
  
  using Solver<TG>::normalize_weights_g;
  using Solver<TG>::normalize_weights_e;
  
  using Solver<TG>::b_0;
  using Solver<TG>::b_e;
  using Solver<TG>::b_g;
  using Solver<TG>::b_gxe;
  using Solver<TG>::delta;
  
  using Solver<TG>::safe_set_g;
  using Solver<TG>::safe_set_gxe;
  using Solver<TG>::safe_set_zero;
  using Solver<TG>::working_set;     

protected:
  MapVec weights_user;
  VecXd xbeta;
  
  double sum_w;
  double sum_E_w;
  double norm2_E_w;
  double denominator_E;
  VecXd norm2_G;
  VecXd norm_G;
  VecXd norm2_GxE;
  VecXd norm_GxE;  
  VecXd G_by_GxE;
  VecXd case1_A22_div_detA;
  VecXd case1_A12_div_detA;  
  VecXd case_3_A;
  VecXd case_3_B;  
  VecXd case_3_E;
  VecXd case_3_F;    
  VecXd case5_A22_div_detA;
  VecXd case5_A12_div_detA;  
  
  double primal_objective;
  VecXd abs_nu_by_G;
  VecXd abs_nu_by_GxE;
  bool abs_nu_by_G_uptodate;
  double x_opt;
  VecXd upperbound_nu_by_G;
  VecXd upperbound_nu_by_GxE;
  
  VecXd abs_inner_nu_by_G;
  VecXd abs_inner_nu_by_GxE;
  
  ArrayXb active_set;
  
  VecXd temp_p;
  VecXd temp_n;
  VecXd weights;
  VecXd Z_w;
  VecXd nu;
  
  double xxx_primal_objective;
  double xxx_primal_objective_1;
  double xxx_primal_objective_2;
  
  public: BinomialSolver(const MapMat& G_,
                         const Eigen::Map<Eigen::VectorXd>& E_,
                 const Eigen::Map<Eigen::VectorXd>& Y_,
                 const Eigen::Map<Eigen::VectorXd>& weights_,
                 bool normalize_) :
    Solver<TG>(G_, E_, Y_, normalize_),
    weights_user(weights_.data(), weights_.rows()),
    xbeta(n),
    norm2_G(p),
    norm_G(p),    
    norm2_GxE(p),
    norm_GxE(p),    
    G_by_GxE(p),
    case1_A22_div_detA(p),
    case1_A12_div_detA(p),
    case_3_A(p),
    case_3_B(p),
    case_3_E(p),
    case_3_F(p),
    case5_A22_div_detA(p),
    case5_A12_div_detA(p),
    abs_nu_by_G(p),
    abs_nu_by_GxE(p),
    upperbound_nu_by_G(p),
    upperbound_nu_by_GxE(p),
    abs_inner_nu_by_G(p),
    abs_inner_nu_by_GxE(p),
    active_set(p),
    temp_p(p),
    temp_n(n),
    weights(n),
    nu(n) {
    
    init();
  }
    
    BinomialSolver(const MapSparseMat& G_,
           const Eigen::Map<Eigen::VectorXd>& E_,
           const Eigen::Map<Eigen::VectorXd>& Y_,
           const Eigen::Map<Eigen::VectorXd>& weights_,
           bool normalize_) :
    Solver<TG>(G_, E_, Y_, normalize_),
    weights_user(weights_.data(), weights_.rows()),
    xbeta(n),
    norm2_G(p),
    norm_G(p),    
    norm2_GxE(p),
    norm_GxE(p),    
    G_by_GxE(p),
    case1_A22_div_detA(p),
    case1_A12_div_detA(p),
    case_3_A(p),
    case_3_B(p),
    case_3_E(p),
    case_3_F(p),
    case5_A22_div_detA(p),
    case5_A12_div_detA(p),
    abs_nu_by_G(p),
    abs_nu_by_GxE(p),
    upperbound_nu_by_G(p),
    upperbound_nu_by_GxE(p),
    abs_inner_nu_by_G(p),
    abs_inner_nu_by_GxE(p),
    active_set(p),
    temp_p(p),
    temp_n(n),
    weights(n),
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
      
      xbeta.setZero(n);
      
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
    
    double update_intercept() {
      xbeta -= normalize_weights_e * E * b_e; 
      xbeta = xbeta.array() - b_0;
      double b_0_old = b_0;
      double b_e_old = b_e;
      double sum_res_w = Z_w.sum() - xbeta.dot(weights);
      b_e = (sum_w * (normalize_weights_e * E.dot(Z_w) - triple_dot_product(E * normalize_weights_e, xbeta, weights)) - 
                        sum_E_w * sum_res_w) / denominator_E;
      
      b_0 = (sum_res_w - sum_E_w * b_e) / sum_w;
      xbeta = xbeta.array() + b_0;
      xbeta += E * b_e * normalize_weights_e;
      double max_diff = std::max(sum_w * sqr(b_0 - b_0_old), norm2_E_w * sqr(b_e_old - b_e));
      if (max_diff > 0) {
        abs_nu_by_G_uptodate = false;
      }
      return max_diff;
    }
    
    double update_b_for_working_set(double lambda_1, double lambda_2, bool active_set_iteration) {
      const double plus_minus_one[] = {-1.0, 1.0};
      double curr_diff;
      double max_diff = update_intercept();
      double G_by_res, GxE_by_res;
      double delta_upperbound, delta_lowerbound;
      bool has_solved;
      int index;
      double b_0_old, b_e_old, b_g_old, b_gxe_old, b_g_new;
      double case1_B1_A22_div_detA;
      double case1_B2, s, root_1, root_2, b_gxe_numerator;
      double case_3_C, case_3_D;
      double s_g, s_gxe, case_3_E_D, case_3_C_F, case_3_B_s_g, root_3, b_g_numerator;
      double case5_B2;

      for (int k = 0; k < working_set.size(); ++k) {
        index = working_set[k];
        if (active_set_iteration && !active_set[index]) {
          continue;
        }
        b_g_old = b_g[index];
        b_gxe_old = b_gxe[index];
        
        xbeta -= normalize_weights_g[index] * (b_g[index] * G.col(index) + normalize_weights_e * G.col(index).cwiseProduct(E) * b_gxe[index]);

        temp_n = Z_w - xbeta.cwiseProduct(weights); // weighted residual
        G_by_res = normalize_weights_g[index] * G.col(index).dot(temp_n);
        GxE_by_res = normalize_weights_e * normalize_weights_g[index] * G.col(index).cwiseProduct(E).dot(temp_n);
        
        if (norm2_GxE[index] == 0.0 || !safe_set_gxe[index]) {
          delta_upperbound = lambda_1 - std::abs(G_by_res);
          delta_lowerbound = std::max(-lambda_2 + std::abs(GxE_by_res), 0.0);
          if (delta_lowerbound <= delta_upperbound) {
            b_g[index] = 0; b_gxe[index] = 0; delta[index] = delta_upperbound;
            curr_diff = norm2_G[index] * sqr(b_g_old);
            max_diff = std::max(max_diff, curr_diff);
            if (curr_diff > 0) {
              abs_nu_by_G_uptodate = false;
              if (!active_set_iteration && !active_set[index]) {
                active_set[index] = true;
              }              
            }
            continue;
          } else {
            b_g_new = soft_threshold(G_by_res, lambda_1) / norm2_G[index];
            b_g[index] = b_g_new; b_gxe[index] = 0; delta[index] = 0;
            xbeta += normalize_weights_g[index] * b_g[index] * G.col(index);
            curr_diff = norm2_G[index] * sqr(b_g_new - b_g_old);
            max_diff = std::max(max_diff, curr_diff);
            if (curr_diff > 0) {
              abs_nu_by_G_uptodate = false;
              if (!active_set_iteration && !active_set[index]) {
                active_set[index] = true;
              }              
            }    
            continue;
          }
        }
        
        delta_upperbound = lambda_1 - std::abs(G_by_res);
        delta_lowerbound = std::max(-lambda_2 + std::abs(GxE_by_res), 0.0);
        if (delta_lowerbound <= delta_upperbound) {
          b_g[index] = 0; b_gxe[index] = 0; delta[index] = delta_upperbound;
          curr_diff = std::max(norm2_G[index] * sqr(b_g_old), 
                               norm2_GxE[index] * sqr(b_gxe_old));
          max_diff = std::max(max_diff, curr_diff);      
          if (curr_diff > 0) {
            abs_nu_by_G_uptodate = false;
            if (!active_set_iteration && !active_set[index]) {
              active_set[index] = true;
            }              
          }
          continue;
        }
        
        has_solved = false;
        // Case 1
        case1_B1_A22_div_detA = G_by_res * case1_A22_div_detA[index];
        for (int i = 0; i < 2; ++i) {
          s = plus_minus_one[i];
          case1_B2 = GxE_by_res - s * (lambda_1 + lambda_2);
          root_1 = case1_B1_A22_div_detA - case1_B2 * case1_A12_div_detA[index];
          root_2 = (case1_B2 - root_1 * G_by_GxE[index]) / norm2_GxE[index];
          if (std::abs(root_2) > std::abs(root_1)) {
            b_gxe_numerator = GxE_by_res - G_by_GxE[index]  * root_1;
            if (s * b_gxe_numerator > lambda_1 + lambda_2) {
              b_g[index] = root_1; b_gxe[index] = root_2; delta[index] = lambda_1;
              xbeta += normalize_weights_g[index] * (b_g[index] * G.col(index) + normalize_weights_e * G.col(index).cwiseProduct(E) * b_gxe[index]);
              curr_diff = std::max(norm2_G[index] * sqr(b_g_old - root_1), 
                                   norm2_GxE[index] * sqr(b_gxe_old - root_2));
              max_diff = std::max(max_diff, curr_diff);              
              if (curr_diff > 0) {
                abs_nu_by_G_uptodate = false;
                if (!active_set_iteration && !active_set[index]) {
                  active_set[index] = true;
                }              
              }     
              has_solved = true;
              break;
            }
          }
        }
        if (has_solved) {
          continue;
        }
        
        // Case 3
        case_3_C = GxE_by_res * G_by_GxE[index] - G_by_res * norm2_GxE[index];
        case_3_D = GxE_by_res * norm2_G[index] - G_by_res * G_by_GxE[index];
        for (int i = 0; i < 2; ++i) {
          s_g = plus_minus_one[i];
          case_3_E_D = s_g * case_3_E[index] + case_3_D;
          case_3_C_F = s_g * case_3_C + case_3_F[index];
          case_3_B_s_g = s_g * 2 * G_by_GxE[index];
          for (int j = 0; j < 2; ++j) {
            s_gxe = plus_minus_one[j];
            root_3 = (s_gxe * case_3_E_D + case_3_C_F) / (case_3_A[index] + s_gxe * case_3_B_s_g);
            if ((root_3 >= 0) && (root_3 < lambda_1)) {
              root_1 = (G_by_res - s_g * (lambda_1 - root_3)) / (norm2_G[index] + s_g * s_gxe * G_by_GxE[index]);
              root_2 = s_g * s_gxe * root_1;
              b_gxe_numerator = GxE_by_res - G_by_GxE[index] * root_1;
              b_g_numerator = (G_by_res - root_2 * G_by_GxE[index]);
              if ((s_gxe * b_gxe_numerator > lambda_2 + root_3) &&
                  (s_g * b_g_numerator > lambda_1 - root_3)) {
                b_g[index] = root_1; b_gxe[index] = root_2; delta[index] = root_3;
                xbeta += normalize_weights_g[index] * (b_g[index] * G.col(index) + normalize_weights_e * G.col(index).cwiseProduct(E) * b_gxe[index]);
                curr_diff = std::max(norm2_G[index] * sqr(b_g_old - root_1), 
                                     norm2_GxE[index] * sqr(b_gxe_old - root_2));
                max_diff = std::max(max_diff, curr_diff);               
                if (curr_diff > 0) {
                  abs_nu_by_G_uptodate = false;
                  if (!active_set_iteration && !active_set[index]) {
                    active_set[index] = true;
                  }              
                }           
                has_solved = true;
                break;
              }          
            }
          }
          if (has_solved) {
            break;
          }      
        }
        if (has_solved) {
          continue;
        }  

        // Case 4
        b_g_new = soft_threshold(G_by_res, lambda_1) / norm2_G[index];
        b_gxe_numerator = GxE_by_res - G_by_GxE[index] * b_g_new;
        if (std::abs(b_gxe_numerator) <= lambda_2) {
          b_g[index] = b_g_new; b_gxe[index] = 0; delta[index] = 0;
          xbeta += normalize_weights_g[index] * b_g[index] * G.col(index);
          curr_diff = std::max(norm2_G[index] * sqr(b_g_old - b_g_new), 
                               norm2_GxE[index] * sqr(b_gxe_old));
          max_diff = std::max(max_diff, curr_diff);         
          if (curr_diff > 0) {
            abs_nu_by_G_uptodate = false;
            if (!active_set_iteration && !active_set[index]) {
              active_set[index] = true;
            }              
          }    
          continue;
        }

        // Case 5
        for (int i = 0; i < 2; ++i) {
          s_g = plus_minus_one[i];
          for (int j = 0; j < 2; ++j) {
            s_gxe = plus_minus_one[j];
            case5_B2 = GxE_by_res - s_gxe * lambda_2;
            root_1 = (G_by_res - s_g * lambda_1) * case5_A22_div_detA[index] - case5_B2 * case5_A12_div_detA[index];
            b_gxe_numerator = GxE_by_res - G_by_GxE[index] * root_1;
            if  (s_gxe * b_gxe_numerator > lambda_2) {
              root_2 = (case5_B2 - root_1 * G_by_GxE[index]) / norm2_GxE[index];
              b_g_numerator = (G_by_res - root_2 * G_by_GxE[index]);

              if (s_g * b_g_numerator > lambda_1) {
                b_g[index] = root_1; b_gxe[index] = root_2; delta[index] = 0;

                xbeta += normalize_weights_g[index] * (b_g[index] * G.col(index) + normalize_weights_e * G.col(index).cwiseProduct(E) * b_gxe[index]);
                curr_diff = std::max(norm2_G[index] * sqr(b_g_old - root_1), 
                                     norm2_GxE[index] * sqr(b_gxe_old - root_2));
                max_diff = std::max(max_diff, curr_diff);                     
                if (curr_diff > 0) {
                  abs_nu_by_G_uptodate = false;
                  if (!active_set_iteration && !active_set[index]) {
                    active_set[index] = true;
                  }
                }
                has_solved = true;
                break;
              }
            }
          }
          if (has_solved) {
            break;
          }      
        }
        if (!has_solved) {
          std::cout << "VERY BAD NEWS\n";
          xbeta += normalize_weights_g[index] * (b_g[index] * G.col(index) + normalize_weights_e * G.col(index).cwiseProduct(E) * b_gxe[index]);
        }
      }
      return max_diff;
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