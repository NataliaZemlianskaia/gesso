#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
 Check these declarations against the C/Fortran source code.
 */

/* .Call calls */
extern SEXP _hierNetGxE_fitModelCVRcpp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _hierNetGxE_fitModelRcpp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"_hierNetGxE_fitModelCVRcpp", (DL_FUNC) &_hierNetGxE_fitModelCVRcpp, 13},
  {"_hierNetGxE_fitModelRcpp",   (DL_FUNC) &_hierNetGxE_fitModelRcpp,   12},
  {NULL, NULL, 0}
};

void R_init_hierNetGxE(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}