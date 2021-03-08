\name{gesso.coefnum}
\alias{gesso.coefnum}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get model coefficients with specified number of non-zero interactions 
}
\description{
Get coefficients with \code{target_b_gxe_non_zero} specified to control the desired sparsity of interactions in the model.
}
\usage{
gesso.coefnum(cv_model, target_b_gxe_non_zero, less_than = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{cv_model}{cross-validated model fit object obtained by using function \code{gesso.cv}}
  \item{target_b_gxe_non_zero}{number of non-zero interactions we want to inlcude in the model}
  \item{less_than}{\code{TRUE} if we want to control a number of \emph{at most} non-zero interactions, \code{FALSE} if we want to control a number of \emph{at least} non-zero interactions}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
A list of model coefficients corresponding to the best model that contains at most or at least \code{target_b_gxe_non_zero} non-zero interaction terms. 

The target model is selected based on the averaged cross-validation (cv) results: for each pair of  parameters \code{lambda}=(lambda_1, lambda_2) in the grid and for each cv fold we obtain a number of non-zero estimated interaction terms, then average cv results by \code{lambda} and choose the tuning parameters corresponding to the minimum average cv loss that have \emph{at most} or \emph{at least} \code{target_b_gxe_non_zero} non-zero interaction terms. Returned coefficients are obtained by fitting the model on the full data with the selected tuning parameters. 

Note that the number of estimated non-zero interactions will only approximately reflect the numbers obtained on cv datasets.
  \item{beta_0 }{estimated intercept value}
  \item{beta_e}{estimated environmental coefficient value}
  \item{beta_g}{a vector of estimated main effect coefficients}
  \item{beta_gxe}{a vector of estimated interaction coefficients}
  \item{beta_c}{a vector of estimated confounders coefficients}
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
model = gesso.cv(data$G_train, data$E_train, data$Y_train)
model_coefficients = gesso.coefnum(model, 5)
gxe_coefficients = model_coefficients$beta_gxe; sum(gxe_coefficients!=0)              
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
