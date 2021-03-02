\name{hierNetGxE.coef}
\alias{hierNetGxE.coef}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get model coefficients
}
\description{
A function to obtain model coefficients from the model fit functions corresponding to the desired pair of tuning paramenters \code{lambda} = (\code{lambda_1}, \code{lambda_2}).
}
\usage{
hierNetGxE.coef(fit, lambda)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{fit}{model fit object obtained either by using function \code{hierNetGxE.fit} or  \code{hierNetGxE.cv}}
  \item{lambda}{a pair of tuning parameters organized in a tibble (ex: \code{lambda = tibble(lambda_1=grid[1], lambda_2=grid[1])})}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
A list of model coefficients corresponding to \code{lambda} values of tuning parameters
  \item{beta_0 }{estimated intercept value}
  \item{beta_e}{estimated environmental coefficient value}
  \item{beta_g}{a vector of estimated main effect coefficients}
  \item{beta_c}{a vector of estimated confounders coefficients}
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
gxe_coefficients = hierNetGxE.coef(model$fit, model$lambda_min)$beta_gxe                
g_coefficients = hierNetGxE.coef(model$fit, model$lambda_min)$beta_g     
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
