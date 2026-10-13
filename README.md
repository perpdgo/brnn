## brnn: Bayesian Regularized Neural Networks for continuous, ordinal and categorical responses

[![CRAN status](https://www.r-pkg.org/badges/version/brnn?color=green)](https://CRAN.R-project.org/package=brnn)
[![CRAN checks](https://badges.cranchecks.info/worst/brnn.svg)](https://cran.r-project.org/web/checks/check_results_brnn.html)
[![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/brnn)](http://www.r-pkg.org/pkg/brnn)
[![Downloads](https://cranlogs.r-pkg.org/badges/brnn?color=blue)](http://www.r-pkg.org/pkg/brnn)

The brnn Package ([Pérez-Rodríguez et al, 2013](https://pubmed.ncbi.nlm.nih.gov/23658327/)) implements Bayesian Regularized Neural Networks 
with a single hidden layer for continuous response. The software is also able to deal ordinal 
responses ([Pérez-Rodríguez et al, 2020](https://acsess.onlinelibrary.wiley.com/doi/full/10.1002/tpg2.20021/)). 
This repository is a copy of the stable version which can be downloaded from [CRAN](https://cran.r-project.org/web/packages/brnn/index.html).

#### Installation

**From CRAN (stable release)**.

```R
install.packages(pkg='brnn',repos='https://cran.r-project.org/')
```

**From GitHub (development version, added features)**.


```R
#1# install remotes
install.packages(pkg='remotes',repos='https://cran.r-project.org/')  
#2# load the library
library(remotes)
#3# install brnn from GitHub
remotes::install_github("perpdgo/brnn")                                                     
```

*Note*: when trying to install from github on a mac you may get the following error message

```
ld: library not found for -lgfortran
```

This can be fixed it by installing gfortan, for mac os you can use [this](https://github.com/fxcoudert/gfortran-for-macOS/releases)

Useful references:

 - [link1](https://yiqingxu.org/public/BigSurError.pdf).
 - [link2](http://thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/).
