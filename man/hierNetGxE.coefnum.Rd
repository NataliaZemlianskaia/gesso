\name{hierNetGxE.coefnum}
\alias{hierNetGxE.coefnum}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get coefficients with \code{target_b_gxe_non_zero} specified to control the desired sparsity of the model.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
hierNetGxE.coefnum(fit, cv_result, target_b_gxe_non_zero)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{fit}{
%%     ~~Describe \code{fit} here~~
}
  \item{cv_result}{
%%     ~~Describe \code{cv_result} here~~
}
  \item{target_b_gxe_non_zero}{
%%     ~~Describe \code{target_b_gxe_non_zero} here~~
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
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (fit, cv_result, target_b_gxe_non_zero) 
{
    best_lambdas = cv_result \%>\% filter(mean_beta_gxe_nonzero <= 
        target_b_gxe_non_zero) \%>\% filter(mean_loss == min(mean_loss)) \%>\% 
        select(lambda_1, lambda_2)
    return(hierNetGxE.coef(fit, best_lambdas))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
