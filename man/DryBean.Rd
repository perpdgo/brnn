\name{DryBean}
\alias{DryBean}
\docType{data}
\title{Dry Bean Classification Data}
\description{
A data frame containing morphological measurements extracted from images
of dry bean grains. The data include 16 numerical predictor variables and
a nominal response variable identifying one of seven bean classes.
}
\usage{
data(DryBean)
}
\format{
A data frame with 13,611 observations and 17 variables:
\describe{
  \item{\code{Area}}{Area of the bean grain measured as the number of pixels within its boundary.}
  \item{\code{Perimeter}}{Length of the bean grain boundary.}
  \item{\code{MajorAxisLength}}{Length of the longest axis of the bean grain.}
  \item{\code{MinorAxisLength}}{Length of the longest axis perpendicular to the major axis.}
  \item{\code{AspectRation}}{Ratio between the major-axis and minor-axis lengths.}
  \item{\code{Eccentricity}}{Eccentricity of the ellipse having the same moments as the bean grain.}
  \item{\code{ConvexArea}}{Area of the smallest convex polygon containing the bean grain.}
  \item{\code{EquivDiameter}}{Diameter of a circle having the same area as the bean grain.}
  \item{\code{Extent}}{Ratio of the bean area to the area of its bounding rectangle.}
  \item{\code{Solidity}}{Ratio of the bean area to its convex area.}
  \item{\code{roundness}}{Measure of the roundness of the bean grain.}
  \item{\code{Compactness}}{Measure of the compactness of the bean grain.}
  \item{\code{ShapeFactor1}}{First dimensionless shape factor.}
  \item{\code{ShapeFactor2}}{Second dimensionless shape factor.}
  \item{\code{ShapeFactor3}}{Third dimensionless shape factor.}
  \item{\code{ShapeFactor4}}{Fourth dimensionless shape factor.}
  \item{\code{Class}}{A factor identifying the bean class: \code{BARBUNYA},
  \code{BOMBAY}, \code{CALI}, \code{DERMASON}, \code{HOROZ},
  \code{SEKER}, or \code{SIRA}.}
}
}
\source{
UCI Machine Learning Repository, Dry Bean Dataset.

Distributed under the Creative Commons Attribution 4.0 International
license:
\url{https://creativecommons.org/licenses/by/4.0/}

For inclusion in this package, the original data were converted to an
R data frame and the variable \code{Class} was stored as a factor.
}

\references{
Koklu, M. and Ozkan, I. A. (2020).
Multiclass classification of dry beans using computer vision and
machine learning techniques.
\emph{Computers and Electronics in Agriculture}, 174, 105507.
\doi{10.1016/j.compag.2020.105507}
}

\keyword{datasets}
