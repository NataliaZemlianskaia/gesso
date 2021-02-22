// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// fitModelCV
Rcpp::List fitModelCV(SEXP G, const Eigen::Map<Eigen::VectorXd>& E, const Eigen::Map<Eigen::VectorXd>& Y, const Eigen::Map<Eigen::MatrixXd>& C, const Rcpp::LogicalVector& normalize, const Eigen::VectorXd& grid, const std::string& family, double tolerance, int max_iterations, int min_working_set_size, int nfolds, int seed, int ncores, int mattype_g);
RcppExport SEXP _hierNetGxE_fitModelCV(SEXP GSEXP, SEXP ESEXP, SEXP YSEXP, SEXP CSEXP, SEXP normalizeSEXP, SEXP gridSEXP, SEXP familySEXP, SEXP toleranceSEXP, SEXP max_iterationsSEXP, SEXP min_working_set_sizeSEXP, SEXP nfoldsSEXP, SEXP seedSEXP, SEXP ncoresSEXP, SEXP mattype_gSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type G(GSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type E(ESEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type C(CSEXP);
    Rcpp::traits::input_parameter< const Rcpp::LogicalVector& >::type normalize(normalizeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type grid(gridSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type family(familySEXP);
    Rcpp::traits::input_parameter< double >::type tolerance(toleranceSEXP);
    Rcpp::traits::input_parameter< int >::type max_iterations(max_iterationsSEXP);
    Rcpp::traits::input_parameter< int >::type min_working_set_size(min_working_set_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type nfolds(nfoldsSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type ncores(ncoresSEXP);
    Rcpp::traits::input_parameter< int >::type mattype_g(mattype_gSEXP);
    rcpp_result_gen = Rcpp::wrap(fitModelCV(G, E, Y, C, normalize, grid, family, tolerance, max_iterations, min_working_set_size, nfolds, seed, ncores, mattype_g));
    return rcpp_result_gen;
END_RCPP
}
// fitModel
Rcpp::List fitModel(SEXP G, const Eigen::Map<Eigen::VectorXd>& E, const Eigen::Map<Eigen::VectorXd>& Y, const Eigen::Map<Eigen::MatrixXd>& C, const Eigen::Map<Eigen::VectorXd>& weights, const Rcpp::LogicalVector& normalize, const Eigen::VectorXd& grid, const std::string& family, double tolerance, int max_iterations, int min_working_set_size, int mattype_g);
RcppExport SEXP _hierNetGxE_fitModel(SEXP GSEXP, SEXP ESEXP, SEXP YSEXP, SEXP CSEXP, SEXP weightsSEXP, SEXP normalizeSEXP, SEXP gridSEXP, SEXP familySEXP, SEXP toleranceSEXP, SEXP max_iterationsSEXP, SEXP min_working_set_sizeSEXP, SEXP mattype_gSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type G(GSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type E(ESEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type C(CSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< const Rcpp::LogicalVector& >::type normalize(normalizeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type grid(gridSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type family(familySEXP);
    Rcpp::traits::input_parameter< double >::type tolerance(toleranceSEXP);
    Rcpp::traits::input_parameter< int >::type max_iterations(max_iterationsSEXP);
    Rcpp::traits::input_parameter< int >::type min_working_set_size(min_working_set_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type mattype_g(mattype_gSEXP);
    rcpp_result_gen = Rcpp::wrap(fitModel(G, E, Y, C, weights, normalize, grid, family, tolerance, max_iterations, min_working_set_size, mattype_g));
    return rcpp_result_gen;
END_RCPP
}
