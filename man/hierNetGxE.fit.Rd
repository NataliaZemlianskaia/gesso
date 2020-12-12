\name{hierNetGxE.fit}
\alias{hierNetGxE.fit}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
hierNetGxE fit
}
\description{
Fits hierNetGxE model over the two dimentional grid of hyperparmeters \code{lambda_1} and \code{lambda_2}, returns estimated coefficients for each pair of hyperparameters.
}
\usage{
hierNetGxE.fit(G, E, Y, normalize = TRUE, grid = NULL, grid_size = 20, 
              grid_min_ratio = 1e-04, family = "gaussian", tolerance = 1e-04, 
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
  \item{tolerance}{tolerance for the dual gap convergence criterion}
  \item{max_iterations}{maximum number of iterations}
  \item{min_working_set_size}{minimum size of the working sets}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
A list of estimated coefficients for fit on each pair of hyperparameters (\code{lambda_1}, \code{lambda_2})
\item{beta_0}{vector of estimated intercept values of size \code{lambda_1}*\code{lambda_2}}
\item{beta_e}{vector of estimated environment coefficients of size \code{lambda_1}*\code{lambda_2}}
\item{beta_g}{matrix of estimated main effects coefficients organized by rows, size (\code{lambda_1}*\code{lambda_2}) by \code{p}}
\item{beta_gxe}{matrix of estimated interactions coefficients organized by rows, size (\code{lambda_1}*\code{lambda_2}) by \code{p}}
\item{num_iterations}{number of iterations until convergence for each fit}
\item{working_set_size}{maximum number of variables in working set for each fit}
\item{num_fitered_by_safe_g}
\item{num_fitered_by_safe_gxe}
\item{objective_value}{objective function (loss) value for each fit}
\item{grid}{vector of values used for hyperparameters tuning}
\item{beta_g_nonzero}{number of estimated non-zero main effects for each fit}
\item{beta_gxe_nonzero}{number of estimated non-zero interactions for each fit}
\item{lambda_1}{\code{lambda_1} path values, decreasing}
\item{lambda_2}{\code{lambda_2} path values, oscillating}
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
data = data.gen()
fit = hierNetGxE.fit(G=data$G_train, E=data$E_train, Y=data$Y_train, normalize=TRUE)
plot(fit$beta_g_nonzero, pch=19, cex=0.4, 
     ylab="num of non-zero features", xlab="lambdas path")
points(fit$beta_gxe_nonzero, pch=19, cex=0.4, col="red")
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
