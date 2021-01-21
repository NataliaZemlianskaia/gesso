#ifndef SOLVER_H
#define SOLVER_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "SolverTypes.h"

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

template<typename A, typename B>
double weighted_squared_norm(const A& x, const B& weights) {
  double result = 0;
  for (int i = 0; i < x.size(); ++i) {
    result += x[i] * x[i] * weights[i];
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

template <typename TG>
class Solver {
  
public:
  const int n;
  const int p;
  TG G;
  MapVec E;
  MapVec Y;
  bool normalize;

  VecXd normalize_weights_g;
  double normalize_weights_e;
  
  double b_0;
  double b_e;
  VecXd b_g;
  VecXd b_gxe;
  VecXd delta;
  
  VecXd xbeta;
  
  ArrayXb safe_set_g;
  ArrayXb safe_set_gxe;
  ArrayXb safe_set_zero;
  std::vector<int> working_set;  
  
  public: Solver(const MapMat& G_,
                 const Eigen::Map<Eigen::VectorXd>& E_,
                 const Eigen::Map<Eigen::VectorXd>& Y_,
                 bool normalize_) :
    n(G_.rows()),
    p(G_.cols()),
    G(G_.data(), G_.rows(), p),
    E(E_.data(), E_.rows()),
    Y(Y_.data(), Y_.rows()),
    normalize(normalize_),
    normalize_weights_g(p),
    b_0(0), 
    b_e(0),
    b_g(p),
    b_gxe(p),
    delta(p),
    xbeta(n),
    safe_set_g(p),
    safe_set_gxe(p),
    safe_set_zero(p) {
    
    base_init();
  }  
    
  Solver(const MapSparseMat& G_,
         const Eigen::Map<Eigen::VectorXd>& E_,
         const Eigen::Map<Eigen::VectorXd>& Y_,
         bool normalize_) :    
    n(G_.rows()),
    p(G_.cols()),
    G(G_),
    E(E_.data(), E_.rows()),
    Y(Y_.data(), Y_.rows()),
    normalize(normalize_),
    normalize_weights_g(p),
    b_0(0), 
    b_e(0),
    b_g(p),
    b_gxe(p),
    delta(p),
    xbeta(n),
    safe_set_g(p),
    safe_set_gxe(p),
    safe_set_zero(p) {

    base_init();
  }     
    
  virtual ~Solver() {}
    
  void base_init() {
    b_g.setZero(p);
    b_gxe.setZero(p);
    delta.setZero(p);
    
    xbeta.setZero(n);
    
    working_set.reserve(p);
  }
  
  virtual int solve(double lambda_1, double lambda_2, double tolerance, int max_iterations, int min_working_set_size) = 0;
    
  double get_b_0() {
    return b_0;
  }
    
  double get_b_e() {
    return b_e * normalize_weights_e;
  }
    
  VecXd get_b_g() {
    return b_g.cwiseProduct(normalize_weights_g);
  }  
    
  VecXd get_b_gxe() {
    return b_gxe.cwiseProduct(normalize_weights_g) * normalize_weights_e;
  }
    
  int get_b_g_non_zero() {
    int result = 0;
    for (int i = 0; i < p; ++i) {
      result += int((b_g[i] != 0) && (normalize_weights_g[i] != 0));
    }
    return result;
  }
    
  int get_b_gxe_non_zero() {
    int result = 0;
    if (normalize_weights_e == 0) {
      return 0;
    }
    for (int i = 0; i < p; ++i) {
      result += int((b_gxe[i] != 0) && (normalize_weights_g[i] != 0));
    }
    return result;
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
    
  virtual double get_value() = 0;

  virtual double get_test_loss(const std::vector<int>& test_idx) = 0;
};

#endif // SOLVER_H
