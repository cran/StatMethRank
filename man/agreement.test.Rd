% Generated by roxygen2 (4.1.0): do not edit by hand
% Please edit documentation in R/agreement.test.R
\name{agreement.test}
\alias{agreement.test}
\title{Test for Agreement of Ranking Data Among Groups}
\usage{
agreement.test(data, method = c("spearman", "kendall"))
}
\arguments{
\item{data}{a data frame of the frequencies of all possible rankings
given by different groups}

\item{method}{whether the test is based on Spearman metric or Kendall metric}
}
\value{
a list of test statistics
}
\description{
This function performs a test of agreement among groups.
}
\examples{
data(Sutton)
agreement.test(Sutton, method = "spearman")
agreement.test(Sutton, method = "kendall")
}
\author{
Li Qinglong <liqinglong0830@163.com>
}
\references{
Intergroup Diversity and Concordance for Ranking Data: An Approach via
Metrics for Permuatations, Paul D. Feigin and Mayer Alvo
}

