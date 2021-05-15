\name{D}
\alias{D}
\title{Genomic dominant relationship matrix for the Jersey dataset.}
\description{

This matrix was calculated by using the dominance incidence matrix derived from 33,267 Single Nucleotide
Polymorphisms (SNPs) information on 297 individually cows, 

\deqn{D=\frac{X_d X_d'}{2 \sum_{j=1}^p (p_j^2+q_j^2) p_j q_j},}

 where

 \itemize{
     \item{\eqn{X_d} is the design matrix for allele substitution effects for dominance.}
     \item{\eqn{p_j} is the frecuency of the second allele at locus \eqn{j} and \eqn{q_j=1-p_j}.}
   }
}

\source{
  University of Wisconsin at Madison, USA.
}

\keyword{datasets}
