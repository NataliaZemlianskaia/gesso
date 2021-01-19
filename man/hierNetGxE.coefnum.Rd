\name{hierNetGxE.coefnum}
\alias{hierNetGxE.coefnum}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get model coefficients with specified number of non-zero interactions 
}
\description{
Get coefficients with \code{target_b_gxe_non_zero} specified to control the desired sparsity of the model interactions.
}
\usage{
hierNetGxE.coefnum(cv_model, target_b_gxe_non_zero)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{cv_model}{cross-validated model fit object obtained by using function \code{hierNetGxE.cv}}
  \item{target_b_gxe_non_zero}{number of \emph{at most} non-zero interactions we want to inlcude in the model}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
A list of model coefficients corresponding to the best model that contains at most \code{target_b_gxe_non_zero} non-zero interaction terms
  \item{beta_0 }{estimated intercept value}
  \item{beta_e}{estimated environmental coefficient value}
  \item{beta_g}{a vector of estimated main effect coefficients}
  \item{beta_gxe}{a vector of estimated interaction coefficients}
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
Natalia Zemlianskaia
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
data = data.gen(sample_size=100, p=500)
model = hierNetGxE.cv(data$G_train, data$E_train, data$Y_train, grid_size=20, 
        parallel=TRUE, nfolds=3)
model_coefficients = hierNetGxE.coefnum(model, 5)
gxe_coefficients = model_coefficients$beta_gxe; sum(gxe_coefficients!=0)              
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
