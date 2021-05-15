/*  brnn/src/util.c by Paulino Perez Rodriguez
 *
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 or 3 of the License
 *  (at your option).
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  A copy of the GNU General Public License is available at
 *  http://www.r-project.org/Licenses/
 */
 
#include "util.h"

/*
 * tansig function, the evaluation of this function 
 * is faster than tanh and the derivative is the same!!! 
*/

double tansig(double x) 
{
    return(2.0/(1.0+exp(-2.0*x)) - 1.0); 
}

/*
 *sech function 
*/
double sech(double x)
{
    return(2.0*exp(x)/(exp(2.0*x)+1.0));
}
 
