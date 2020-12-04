\name{data.gen}
\alias{data.gen}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Data Generation.
}
\description{
Generates genotypes data matrix G (\code{sample_size} by \code{p}), vector of environmental measurments E, and an outcome vector Y of size \code{sample_size}. Simulates training, validation, and test datasets.
}
\usage{
data.gen(sample_size = 100, p = 20, n_g_non_zero = 15, n_gxe_non_zero = 10, 
        family = "gaussian", mode = "strong_hierarchical", normalize = TRUE, 
        seed = 1, pG = 0.2, pE = 0.3)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{sample_size}{sample size of the data}
  \item{p}{total number of main effects}
  \item{n_g_non_zero}{number of non-zero main effects to generate}
  \item{n_gxe_non_zero}{number of non-zero interactions effects to generate}
  \item{family}{"gaussian" for continous Y and "binomial" for binary}
  \item{mode}{either "strong_hierarchical", "hierarchical", or "anti-hierarchical"}
  \item{normalize}{\code{TRUE} to normalize matrix \code{G} and vector \code{E}}
  \item{seed}{random seed}
  \item{pG}{genotypes prevalence, value from 0 to 1}
  \item{pE}{environment prevalence, value from 0 to 1}
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
G = data$G_train; GxE = data$GxE_train
E = data$E_train; Y = data$Y_train
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
