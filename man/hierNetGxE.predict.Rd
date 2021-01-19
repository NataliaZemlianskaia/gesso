\name{hierNetGxE.predict}
\alias{hierNetGxE.predict}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Predict new outcome vector
}
\description{
Predict new outcome vector based on the new data and estimated model coefficients.
}
\usage{
hierNetGxE.predict(beta_0, beta_e, beta_g, beta_gxe, new_G, new_E, family = "gaussian")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{beta_0 }{estimated intercept value}
  \item{beta_e}{estimated environmental coefficient value}
  \item{beta_g}{a vector of estimated main effect coefficients}
  \item{beta_gxe}{a vector of estimated interaction coefficients}
  \item{new_G}{matrix of main effects, predictors organized by columns}
  \item{new_E}{vector of environmental measurments}
  \item{family}{set \code{family="gaussian"} for the continuous outcome and 
          \code{family="binomial"} for the binary outcome with 0/1 levels}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
Returns a vector of predicted values
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
data = data.gen(sample_size=100, p=200)
tune_model = hierNetGxE.cv(data$G_train, data$E_train, data$Y_train)
coefficients = hierNetGxE.coef(tune_model$fit, tune_model$lambda_min)
beta_0 = coefficients$beta_0; beta_e = coefficients$beta_e                   
beta_g = coefficients$beta_g; beta_gxe = coefficients$beta_gxe     

new_G = data$G_test; new_E = data$E_test
new_Y = hierNetGxE.predict(beta_0, beta_e, beta_g, beta_gxe, new_G, new_E)
cor(new_Y, data$Y_test)^2
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
