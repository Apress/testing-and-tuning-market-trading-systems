/******************************************************************************/
/*                                                                            */
/*  RANDOM - Assorted non-uniform random number generators.                   */
/*           They all call an external uniform generator, unifrand().         */
/*                                                                            */
/*    normal () - Normal (mean zero, unit variance)                           */
/*    normal_pair ( double *x1 , double *x2 ) - Pair of standard normals      */
/*    beta ( int v1 , int v2 ) - Beta with parameters v1 / 2 and v2 / v2      */
/*    rand_sphere ( int nvars , double *x ) - Uniform on unit sphere surface  */
/*    cauchy ( int n , double scale , double *x ) - Multivariate Cauchy       */
/*                                                                            */
/******************************************************************************/

#include <math.h>

double unifrand () ;

#if ! defined ( PI )
#define PI 3.141592653589793
#endif

/*
--------------------------------------------------------------------------------

   Generate a standard normal random variable or a pair of them
   using the Box-Muller method.

--------------------------------------------------------------------------------
*/

double normal ()
{
   double x1, x2 ;

   for (;;) {
      x1 = unifrand () ;
      if (x1 <= 0.0)      // Safety: log(0) is undefined
         continue ;
      x1 = sqrt ( -2.0 * log ( x1 )) ;
      x2 = cos ( 2.0 * PI * unifrand () ) ;
      return x1 * x2 ;
      }
}

void normal_pair ( double *x1 , double *x2 )
{
   double u1, u2 ;

   for (;;) {
      u1 = unifrand () ;
      if (u1 <= 0.0)      // Safety: log(0) is undefined
         continue ;
      u1 = sqrt ( -2.0 * log ( u1 )) ;
      u2 = 2.0 * PI * unifrand () ;
      *x1 = u1 * sin ( u2 ) ;
      *x2 = u1 * cos ( u2 ) ;
      return ;
      }
}

/*
--------------------------------------------------------------------------------

   Generate a Gamma random variable having parameter v/2

--------------------------------------------------------------------------------
*/

double gamma ( int v )
{
   double x, y, z, vm1, root ;

   switch (v) {

      case 1:                   // Chi-square with 1 df is 2 gamma(.5)
         x = normal () ;
         return 0.5 * x * x ;

      case 2:                   // Gamma(1) is exponential(1)
         for (;;) {
            x = unifrand () ;
            if (x > 0.0)
               return -log ( x ) ;
            }

      default:                  // Valid for all real a>1 (a=v/2)
         vm1 = 0.5 * v - 1.0 ;
         root = sqrt ( v - 1.0 ) ;

         for (;;) {
            y = tan ( PI * unifrand () ) ;
            x = root * y + vm1 ;
            if (x <= 0.0)
               continue ;
            z = (1.0 + y * y) * exp ( vm1 * log(x/vm1) - root * y ) ;
            if (unifrand () <= z)
               return x ;
            }
      }
}


/*
--------------------------------------------------------------------------------

   Generate a beta random variable with parameters v1 / 2 and v2 / 2.

--------------------------------------------------------------------------------
*/

double beta ( int v1 , int v2 )
{
   double x1, x2 ;

   x1 = gamma ( v1 ) ;
   x2 = gamma ( v2 ) ;

   return x1 / (x1 + x2) ;
}

/*
--------------------------------------------------------------------------------

   Generate a random point on an n-sphere.

--------------------------------------------------------------------------------
*/

void rand_sphere ( int nvars , double *x )
{
   int i ;
   double length ;

   length = 0.0 ;
   for (i=0 ; i<nvars/2 ; i++) {      // Efficiently generate pairs
      normal_pair ( &x[2*i] , &x[2*i+1] ) ;
      length += x[2*i] * x[2*i]  +  x[2*i+1] * x[2*i+1] ;
      }

   if (nvars % 2) {                   // If odd, get the last one
      x[nvars-1] = normal () ;
      length += x[nvars-1] * x[nvars-1] ;
      }

   length = 1.0 / sqrt ( length ) ;
   while (nvars--)
      x[nvars] *= length ;
}


/*
--------------------------------------------------------------------------------

   Generate a random vector following the n-variate Cauchy density
   with specified scale.

--------------------------------------------------------------------------------
*/

void cauchy ( int n , double scale , double *x )
{
   double temp ;

   if (n == 1) {
      temp = PI * unifrand () - 0.5 * PI ;
      x[0] = scale * tan ( 0.99999999 * temp ) ;
      return ;
      }

   rand_sphere ( n , x ) ;

   temp = beta ( n , 1 ) ;

   if (temp < 1.0)
      temp = scale * sqrt ( temp / (1.0 - temp) ) ;
   else
      temp = 1.e10 ;

   while (n--)
      x[n] *= temp ;
}

