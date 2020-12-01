\name{hierNetGxE.fit}
\alias{hierNetGxE.fit}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
hierNetGxE fit
}
\description{
Fits hierNetGxE model over the two dimentional grid of hyperparmeters
}
\usage{
hierNetGxE.fit(G, E, Y, normalize = FALSE, grid = NULL, grid_size = 20, 
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
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
