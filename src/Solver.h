#ifndef SOLVER_H
#define SOLVER_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

inline double sqr(double x) {
  return x * x;
}

inline double sign(double x) {
  return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

inline double soft_threshold(double x, double lambda) {
  if (x > lambda) {
    return x - lambda;
  }
  if (x < - lambda){
    return x + lambda;
  }
  return 0;
}

template<typename A, typename B, typename C>
double triple_dot_product(const A& a, const B& b, const C& c) {
  double result = 0;
  for (int i = 0; i < a.size(); ++i) {
    result += a[i] * b[i] * c[i];
  }
  return result;
}

template<typename T>
std::vector<int> argsort(const T& array) {
  std::vector<int> indices(array.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&array](int left, int right) -> bool {
              // sort indices according to corresponding array element
              return array[left] < array[right];
            });
  
  return indices;
}

class Solver {
  typedef Eigen::Map<const Eigen::MatrixXd> MapMat;
  typedef Eigen::Map<const Eigen::VectorXd> MapVec;
  typedef Eigen::VectorXd VecXd;
  typedef Eigen::VectorXi VecXi;
  typedef Eigen::ArrayXd ArrayXd;
  typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb;
  
protected:
  const int p;
  MapMat G;
  MapVec E;
  MapVec Y;
  VecXd res;
  VecXd res_weights_sqrt;
  MapVec weights;
  VecXd weights_sqrt;
  
  double b_0;
  double b_e;
  VecXd b_g;
  VecXd b_gxe;
  VecXd delta;
  
  double sum_w;
  double sum_E_w;
  double norm2_E_w;
  double denominator_E;
  VecXd norm2_G;
  VecXd norm_G;
  VecXd norm2_GxE;
  VecXd norm_GxE;  
  VecXd G_by_GxE;
  double norm2_Y_w;
  VecXd case1_A22_div_detA;
  VecXd case1_A12_div_detA;  
  VecXd case_3_A;
  VecXd case_3_B;  
  VecXd case_3_E;
  VecXd case_3_F;    
  VecXd case5_A22_div_detA;
  VecXd case5_A12_div_detA;  
  
  VecXd nu;
  double primal_objective;
  double dual_objective;
  VecXd abs_nu_by_G;
  VecXd abs_nu_by_GxE;
  
  ArrayXb safe_set_g;
  ArrayXb safe_set_gxe;
  ArrayXb safe_set_zero;
  std::vector<int> working_set;
  
  VecXd inner_nu;
  double inner_dual_objective;
  VecXd abs_inner_nu_by_G;
  VecXd abs_inner_nu_by_GxE;
  
  ArrayXb active_set;
  
  public: Solver(const Eigen::Map<Eigen::MatrixXd>& G_,
                 const Eigen::Map<Eigen::VectorXd>& E_,
                 const Eigen::Map<Eigen::VectorXd>& Y_,
                 const Eigen::Map<Eigen::VectorXd>& weights_) :
    p(G_.cols()),
    G(G_.data(), G_.rows(), p),
    E(E_.data(), E_.rows()),
    Y(Y_.data(), Y_.rows()),
    weights(weights_.data(), weights_.rows()),
    b_0(0), 
    b_e(0),
    norm2_G(p),
    norm2_GxE(p),
    G_by_GxE(p),
    nu(G_.rows()),
    abs_nu_by_G(p),
    abs_nu_by_GxE(p),
    safe_set_g(p),
    safe_set_gxe(p),
    safe_set_zero(p),
    inner_nu(G_.rows()),
    abs_inner_nu_by_G(p),
    abs_inner_nu_by_GxE(p),
    active_set(p) {
    
    b_g.setZero(p);
    b_gxe.setZero(p);
    delta.setZero(p);
    
    working_set.reserve(p);
    
    update_weighted_variables();
  }
    
    void update_weighted_variables() {
      weights_sqrt = weights.cwiseSqrt();
      
      sum_w = weights.sum();
      sum_E_w = E.dot(weights);
      norm2_E_w = E.cwiseProduct(E).dot(weights);
      denominator_E = sum_w * norm2_E_w - sqr(sum_E_w);
      
      VecXd temp;
      for (int i = 0; i < G.cols(); ++i) {
        temp = G.col(i).cwiseProduct(G.col(i));
        norm2_G[i] = temp.dot(weights);
        temp = temp.cwiseProduct(E);
        G_by_GxE[i] = temp.dot(weights);
        temp = temp.cwiseProduct(E);
        norm2_GxE[i] = temp.dot(weights);
      }
      
      norm_G = norm2_G.cwiseSqrt();
      norm_GxE = norm2_GxE.cwiseSqrt();
      
      const VecXd case1_detA = norm2_G.cwiseProduct(norm2_GxE) - G_by_GxE.cwiseProduct(G_by_GxE);
      case1_A22_div_detA = norm2_GxE.cwiseQuotient(case1_detA);
      case1_A12_div_detA = G_by_GxE.cwiseQuotient(case1_detA);
      case_3_A = (norm2_G + norm2_GxE);
      case_3_B = 2 * G_by_GxE;
      const VecXd case5_detA = (norm2_G.cwiseProduct(norm2_GxE) - G_by_GxE.cwiseProduct(G_by_GxE));
      case5_A22_div_detA = norm2_GxE.cwiseQuotient(case5_detA);
      case5_A12_div_detA = G_by_GxE.cwiseQuotient(case5_detA); 
      
      norm2_Y_w = Y.cwiseProduct(Y).dot(weights);
      
      res = ((Y.array() - b_0).matrix() - G * (b_g + E.cwiseProduct(b_gxe)) - E * b_e).cwiseProduct(weights);
      res_weights_sqrt = ((Y.array() - b_0).matrix() - G * (b_g + E.cwiseProduct(b_gxe)) - E * b_e).cwiseProduct(weights_sqrt);
    }
    
    int solve(double lambda_1, double lambda_2, double tolerance, int max_iterations, int min_working_set_size) {
      dual_objective = -std::numeric_limits<double>::infinity();
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
      
      case_3_E = G_by_GxE * (lambda_1 - lambda_2);
      case_3_F = (lambda_1 * norm2_GxE - lambda_2 * norm2_G);
      
      double duality_gap, inner_duality_gap, max_diff_tolerance, max_diff;
      while (num_passes < max_iterations) {
        duality_gap = check_duality_gap(lambda_1, lambda_2, false);
        if (duality_gap < tolerance) {
          break;
        }
        
        update_working_set(lambda_1, lambda_2, duality_gap, working_set_size);
        working_set_size = std::min(2 * working_set_size, p);
        
        inner_dual_objective = -std::numeric_limits<double>::infinity();
        active_set.setZero(p);
        max_diff_tolerance = tolerance * 10;
        while (num_passes < max_iterations) {
          inner_duality_gap = check_duality_gap(lambda_1, lambda_2, true);
          if (inner_duality_gap < tolerance) {
            break;
          } else {
            max_diff_tolerance /= 10;
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
    
    double compute_dual_objective(const Eigen::Ref<VecXd>& nu) {
      return (norm2_Y_w - (Y.cwiseProduct(weights_sqrt) - nu).squaredNorm()) / 2.0;
    }
    
    double naive_projection(double lambda_1, double lambda_2, const Eigen::Ref<VecXd>& nu, const Eigen::Ref<VecXd>& abs_nu_by_G, const Eigen::Ref<VecXd>& abs_nu_by_GxE) {
      VecXd dual_delta = (lambda_1 * abs_nu_by_GxE - lambda_2 * abs_nu_by_G).cwiseQuotient(abs_nu_by_GxE + abs_nu_by_G).cwiseMax(0).cwiseMin(lambda_1);
      double M = std::numeric_limits<double>::infinity();
      for (int i = 0; i < abs_nu_by_G.size(); ++i) {
        if (abs_nu_by_G[i] > 0) {
          M = std::min(M, (lambda_1 - dual_delta[i]) / abs_nu_by_G[i]);
        }
        if (abs_nu_by_GxE[i] > 0) {
          M = std::min(M, (lambda_2 + dual_delta[i]) / abs_nu_by_GxE[i]);
        }      
      }
      double x_hat = Y.cwiseProduct(weights_sqrt).dot(nu) / nu.squaredNorm();
      double x_opt;
      if (std::abs(x_hat) <= M) {
        x_opt = x_hat;
      } else {
        x_opt = sign(x_hat) * M;
      }
      return x_opt;
    }
    
    // updates abs_nu_by_G, abs_nu_by_GxE 
    // dual_objective and nu.
    double update_nu(double lambda_1, double lambda_2) {
      VecXd nu_res = res_weights_sqrt;
      abs_nu_by_G = (res.transpose() * G).cwiseAbs().transpose();
      abs_nu_by_GxE = (res.cwiseProduct(E).transpose() * G).cwiseAbs().transpose();
      double x_opt = naive_projection(lambda_1, lambda_2, nu_res, abs_nu_by_G, abs_nu_by_GxE);
      nu_res *= x_opt;
      double dual_objective_res = compute_dual_objective(nu_res);
      if (dual_objective_res > dual_objective) {
        dual_objective = dual_objective_res;
        nu.swap(nu_res);
        abs_nu_by_G *= x_opt;
        abs_nu_by_GxE *= x_opt;
      } else {
        // Rare case -- new dual variable nu_res is apparently worse than a previous one.
        abs_nu_by_G = (nu.cwiseProduct(weights_sqrt).transpose() * G).cwiseAbs().transpose();
        abs_nu_by_GxE = (nu.cwiseProduct(weights_sqrt).cwiseProduct(E).transpose() * G).cwiseAbs().transpose();
      }
      return dual_objective;
    }
    
    double update_inner_nu(double lambda_1, double lambda_2) {
      VecXd inner_nu_res = res_weights_sqrt;
      //    abs_inner_nu_by_G = (inner_nu_res * G(Eigen::all, working_set)).cwiseAbs();
      //    abs_inner_nu_by_GxE = (inner_nu_res.cwiseProduct(E) * G(Eigen::all, working_set)).cwiseAbs();
      // For details on resize vs conservativeResize see
      // https://stackoverflow.com/questions/34449805/reserve-dense-eigen-matrix
      //abs_inner_nu_by_G.conservativeResize(working_set.size());
      //abs_inner_nu_by_GxE.conservativeResize(working_set.size());
      abs_inner_nu_by_G.setZero(working_set.size());
      abs_inner_nu_by_GxE.setZero(working_set.size());
      for (int i = 0; i < working_set.size(); ++i) {
        abs_inner_nu_by_G[i] = std::abs(res.dot(G.col(working_set[i])));
        abs_inner_nu_by_GxE[i] = std::abs(res.cwiseProduct(E).dot(G.col(working_set[i])));
      }
      double x_opt = naive_projection(lambda_1, lambda_2, inner_nu_res, abs_inner_nu_by_G, abs_inner_nu_by_GxE);
      inner_nu_res *= x_opt;
      double inner_dual_objective_res = compute_dual_objective(inner_nu_res);
      if (inner_dual_objective_res > inner_dual_objective) {
        inner_dual_objective = inner_dual_objective_res;
        inner_nu.swap(inner_nu_res);
      }
      return inner_dual_objective;
    }
    
    double check_duality_gap(double lambda_1, double lambda_2, bool use_working_set) {
      update_intercept();
      double current_dual_objective;
      if (use_working_set) {
        current_dual_objective = update_inner_nu(lambda_1, lambda_2);  
      } else {
        current_dual_objective = update_nu(lambda_1, lambda_2);  
      }
      primal_objective = res_weights_sqrt.squaredNorm() / 2.0 + lambda_1 * (b_g.cwiseAbs().cwiseMax(b_gxe.cwiseAbs())).sum() + lambda_2 * b_gxe.cwiseAbs().sum();
      return primal_objective - current_dual_objective;
    }
    
    void update_working_set(double lambda_1, double lambda_2, double dual_gap, int working_set_size) {
      double r = sqrt(2 * dual_gap);
      ArrayXd d_j = ((lambda_1 - abs_nu_by_G.array()) + (r * norm_GxE).array().max(lambda_2 - abs_nu_by_GxE.array())) / (norm_GxE + norm_G).array();
      //ArrayXd d_j = (lambda_1 - lambda_2 - abs_nu_by_G.array() - (lambda_2 - r * norm_GxE.array()).max(abs_nu_by_GxE.array())) / (norm_GxE + norm_G).array();
      
      VecXd upperbound_nu_by_G = abs_nu_by_G + r * norm_G;
      VecXd upperbound_nu_by_GxE = abs_nu_by_GxE + r * norm_GxE;
      safe_set_zero = (upperbound_nu_by_GxE.array() - lambda_2).max(0) < (lambda_1 - upperbound_nu_by_G.array());
      for (int i = 0; i < p; ++i) {
        safe_set_gxe[i] = safe_set_gxe[i] && (!safe_set_zero[i]) && (upperbound_nu_by_GxE[i] >= lambda_2);
        safe_set_g[i] = safe_set_g[i] && (!safe_set_zero[i]) && (upperbound_nu_by_G[i] >= lambda_1 || safe_set_gxe[i]);      
      }
      
      for (int i = 0; i < p; ++i) {
        if (b_gxe[i] != 0 && (!safe_set_gxe[i])) {
          res += (G.col(i).cwiseProduct(E) * b_gxe[i]).cwiseProduct(weights);
          res_weights_sqrt += (G.col(i).cwiseProduct(E) * b_gxe[i]).cwiseProduct(weights_sqrt);
          b_gxe[i] = 0;
        }
        if (b_g[i] != 0 && (!safe_set_g[i])) {
          res += (G.col(i) * b_g[i]).cwiseProduct(weights);
          res_weights_sqrt += (G.col(i) * b_g[i]).cwiseProduct(weights_sqrt);
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
      sort(working_set.begin(), working_set.end());
    }
    
    double update_intercept() {
      res += E.cwiseProduct(weights) * b_e;
      res += weights * b_0;
      res_weights_sqrt += weights_sqrt * b_0;
      res_weights_sqrt += E.cwiseProduct(weights_sqrt) * b_e;      
      double b_0_old = b_0;
      double b_e_old = b_e;
      double sum_res = res.sum();
      b_e = (sum_w * E.dot(res) - sum_E_w * sum_res) / denominator_E;
      b_0 = (sum_res - sum_E_w * b_e) / sum_w;
      res -= weights * b_0;
      res -= E.cwiseProduct(weights) * b_e;
      res_weights_sqrt -= weights_sqrt * b_0;
      res_weights_sqrt -= E.cwiseProduct(weights_sqrt) * b_e;
      return std::max(sum_w * sqr(b_0 - b_0_old), norm2_E_w * sqr(b_e_old - b_e));
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
        
        res += (b_g[index] * G.col(index) + G.col(index).cwiseProduct(E) * b_gxe[index]).cwiseProduct(weights);
        res_weights_sqrt += (b_g[index] * G.col(index) + G.col(index).cwiseProduct(E) * b_gxe[index]).cwiseProduct(weights_sqrt);
        
        G_by_res = G.col(index).dot(res);
        GxE_by_res = G.col(index).dot(res.cwiseProduct(E));
        
        if (norm2_GxE[index] == 0.0 || !safe_set_gxe[index]) {
          delta_upperbound = lambda_1 - std::abs(G_by_res);
          delta_lowerbound = std::max(-lambda_2 + std::abs(GxE_by_res), 0.0);
          if (delta_lowerbound <= delta_upperbound) {
            b_g[index] = 0; b_gxe[index] = 0; delta[index] = delta_upperbound;
            curr_diff = norm2_G[index] * sqr(b_g_old);
            max_diff = std::max(max_diff, curr_diff);
            if (!active_set_iteration && !active_set[index]) {
              active_set[index] = int(curr_diff > 0);
            }
            continue;
          } else {
            b_g_new = soft_threshold(G_by_res, lambda_1) / norm2_G[index];
            b_g[index] = b_g_new; b_gxe[index] = 0; delta[index] = 0;
            res -= (b_g[index] * G.col(index)).cwiseProduct(weights);
            res_weights_sqrt -= (b_g[index] * G.col(index)).cwiseProduct(weights_sqrt);
            curr_diff = norm2_G[index] * sqr(b_g_new - b_g_old);
            max_diff = std::max(max_diff, curr_diff);
            if (!active_set_iteration && !active_set[index]) {
              active_set[index] = int(curr_diff > 0);
            }        
            continue;
          }
        }
        
        
        // Case 2
        delta_upperbound = lambda_1 - std::abs(G_by_res);
        delta_lowerbound = std::max(-lambda_2 + std::abs(GxE_by_res), 0.0);
        if (delta_lowerbound <= delta_upperbound) {
          b_g[index] = 0; b_gxe[index] = 0; delta[index] = delta_upperbound;
          curr_diff = std::max(norm2_G[index] * sqr(b_g_old), 
                               norm2_GxE[index] * sqr(b_gxe_old));
          max_diff = std::max(max_diff, curr_diff);      
          if (!active_set_iteration && !active_set[index]) {
            active_set[index] = int(curr_diff > 0);
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
              res -= (b_g[index] * G.col(index) + G.col(index).cwiseProduct(E) * b_gxe[index]).cwiseProduct(weights);
              res_weights_sqrt -= (b_g[index] * G.col(index) + G.col(index).cwiseProduct(E) * b_gxe[index]).cwiseProduct(weights_sqrt);
              curr_diff = std::max(norm2_G[index] * sqr(b_g_old - root_1), 
                                   norm2_GxE[index] * sqr(b_gxe_old - root_2));
              max_diff = std::max(max_diff, curr_diff);              
              if (!active_set_iteration && !active_set[index]) {
                active_set[index] = int(curr_diff > 0);
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
                res -= (b_g[index] * G.col(index) + G.col(index).cwiseProduct(E) * b_gxe[index]).cwiseProduct(weights);
                res_weights_sqrt -= (b_g[index] * G.col(index) + G.col(index).cwiseProduct(E) * b_gxe[index]).cwiseProduct(weights_sqrt);
                curr_diff = std::max(norm2_G[index] * sqr(b_g_old - root_1), 
                                     norm2_GxE[index] * sqr(b_gxe_old - root_2));
                max_diff = std::max(max_diff, curr_diff);               
                if (!active_set_iteration && !active_set[index]) {
                  active_set[index] = int(curr_diff > 0);
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
          res -= (b_g[index] * G.col(index)).cwiseProduct(weights);
          res_weights_sqrt -= (b_g[index] * G.col(index)).cwiseProduct(weights_sqrt);
          curr_diff = std::max(norm2_G[index] * sqr(b_g_old - b_g_new), 
                               norm2_GxE[index] * sqr(b_gxe_old));
          max_diff = std::max(max_diff, curr_diff);         
          if (!active_set_iteration && !active_set[index]) {
            active_set[index] = int(curr_diff > 0);
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
                res -= (b_g[index] * G.col(index) + G.col(index).cwiseProduct(E) * b_gxe[index]).cwiseProduct(weights);
                res_weights_sqrt -= (b_g[index] * G.col(index) + G.col(index).cwiseProduct(E) * b_gxe[index]).cwiseProduct(weights_sqrt);
                curr_diff = std::max(norm2_G[index] * sqr(b_g_old - root_1), 
                                     norm2_GxE[index] * sqr(b_gxe_old - root_2));
                max_diff = std::max(max_diff, curr_diff);                     
                if (!active_set_iteration && !active_set[index]) {
                  active_set[index] = int(curr_diff > 0);
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
      }
      return max_diff;
    }
    
    double get_b_0() {
      return b_0;
    }
    
    double get_b_e() {
      return b_e;
    }
    
    VecXd get_b_g() {
      return b_g;
    }  
    
    VecXd get_b_gxe() {
      return b_gxe;
    }
    
    int get_working_set_size() {
      return working_set.size();
    }
    
    int get_num_fitered_by_safe_g() {
      int num_filtered = 0;
      for (int i = 0; i < safe_set_g.size(); ++i) {
        if (!safe_set_g[i]) {
          ++num_filtered;
        }
      }
      return num_filtered;
    }
    
    int get_num_fitered_by_safe_gxe() {
      int num_filtered = 0;
      for (int i = 0; i < safe_set_g.size(); ++i) {
        if (!safe_set_g[i]) {
          ++num_filtered;
        }
      }
      return num_filtered;
    }
    
    double get_value() {
      return primal_objective;
    }

    double get_test_loss(const Eigen::Map<Eigen::VectorXi>& test_idx) {
      double test_loss = 0;
      double residual;
      int index;
      for (int i = 0; i < test_idx.size(); ++i) {
        index = test_idx[i];
        residual = Y[index] - G.row(index).dot(b_g + b_gxe * E[index]) - b_0 - b_e * E[index];
        test_loss += residual * residual;
      }
      return test_loss;
    }
    
};

#endif // SOLVER_H
