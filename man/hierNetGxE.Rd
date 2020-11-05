\name{hierNetGxE}
\alias{hierNetGxE}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
HierNet GxE Model Fit
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
hierNetGxE(G, E, Y, standardize = FALSE, grid = NULL, grid_size = 10, grid_min_ratio = 1e-04, family = "gaussian", tolerance = 1e-04, max_iterations = 10000, min_working_set_size = 100)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{G}{
%%     ~~Describe \code{G} here~~
}
  \item{E}{
%%     ~~Describe \code{E} here~~
}
  \item{Y}{
%%     ~~Describe \code{Y} here~~
}
  \item{standardize}{
%%     ~~Describe \code{standardize} here~~
}
  \item{grid}{
%%     ~~Describe \code{grid} here~~
}
  \item{grid_size}{
%%     ~~Describe \code{grid_size} here~~
}
  \item{grid_min_ratio}{
%%     ~~Describe \code{grid_min_ratio} here~~
}
  \item{family}{
%%     ~~Describe \code{family} here~~
}
  \item{tolerance}{
%%     ~~Describe \code{tolerance} here~~
}
  \item{max_iterations}{
%%     ~~Describe \code{max_iterations} here~~
}
  \item{min_working_set_size}{
%%     ~~Describe \code{min_working_set_size} here~~
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
function (G, E, Y, standardize = FALSE, grid = NULL, grid_size = 10, 
    grid_min_ratio = 1e-04, family = "gaussian", tolerance = 1e-04, 
    max_iterations = 10000, min_working_set_size = 100) 
{
    if (is.null(grid)) {
        grid = 10^seq(-4, log10(0.1), length.out = grid_size)
    }
    n = dim(data$G_train)[1]
    weights = rep(1, n)/n
    return(fitModelRcpp(G, E, Y, weights, standardize, grid, 
        grid_size, grid_min_ratio, family, tolerance, max_iterations, 
        min_working_set_size))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
