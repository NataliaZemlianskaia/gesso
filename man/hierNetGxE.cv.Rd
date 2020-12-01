\name{hierNetGxE.cv}
\alias{hierNetGxE.cv}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Cross-Validation
}
\description{
Performs \code{nfolds}-fold cross-validation for hierNetGxE
}
\usage{
hierNetGxE.cv(G, E, Y, normalize = FALSE, grid = NULL, grid_size = 20, 
              grid_min_ratio = 1e-04, family = "gaussian", nfolds = 5, parallel = TRUE, 
              seed = 42, tolerance = 1e-04, 
              max_iterations = 10000, min_working_set_size = 100)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{G}{matrix of main effects}
  \item{E}{vector of environmental measurments}
  \item{Y}{outcome vector}
  \item{normalize}{\code{TRUE} to normalize matrix \code{G} and vector \code{E}}
  \item{grid}{grid sequence for tuning hyperparameters}
  \item{grid_size}{specify grid size to generate grid automatically}
  \item{grid_min_ratio}{}
  \item{family}{"gaussian" for continuous outcome and "binomial" for binary}
  \item{nfolds}{number of cross-validation splits}
  \item{parallel}{\code{TRUE} to enable parallel cross-validation}
  \item{seed}{set random seed to control random folds assignment}
  \item{tolerance}{tolerance for the dual gap convergence criterion}
  \item{max_iterations}{maximum number of iterations}
  \item{min_working_set_size}{minimum size of the working sets}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
data = data.gen(sample_size=1000, p=100000)
model = hierNetGxE.cv(data$G_train, data$E_train, data$Y_train, grid_size=20, 
        parallel=TRUE, nfolds=3)
gxe_coefficients = hierNetGxE.coef(model$fit, model$lambda.min)$beta_gxe                      
g_coefficients = hierNetGxE.coef(model$fit, model$lambda.min)$beta_g                      

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
