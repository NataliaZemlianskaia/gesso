\name{hierNetGxE.coef}
\alias{hierNetGxE.coef}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get coefficients.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
hierNetGxE.coef(fit, lambda)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{fit}{
%%     ~~Describe \code{fit} here~~
}
  \item{lambda}{
%%     ~~Describe \code{lambda} here~~
}
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
