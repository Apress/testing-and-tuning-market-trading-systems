/******************************************************************************/
/*                                                                            */
/*  BRENTMAX - Use Brent's method to find a local maximum of a                */
/*             univariate function.                                           */
/*                                                                            */
/*  This is given three points such that the center has greater function      */
/*  value than its neighbors.  It iteratively refines the interval.           */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "headers.h"

#define DEBUG 0

double brentmax (
   int itmax ,            // Iteration limit
   double eps ,           // Function convergence tolerance
   double tol ,           // X convergence tolerance
   double (*c_func) (double) , // Criterion function
   double *xa ,           // Lower X value, input and output
   double *xb ,           // Middle (best), input and output
   double *xc ,           // And upper, input and output
   double y               // Function value at xb
   )
{
   int iter ;
   double x0, x1, x2, y0, y1, y2, xleft, xmid, xright, movement, trial ;
   double small_step, small_dist, numer, denom, temp1, temp2 ;
   double testdist, this_x, this_y ;

/*
   Initialize
*/

   x0 = x1 = x2 = *xb ;
   xleft = *xa ;
   xright = *xc ;

   y0 = y1 = y2 = y ;

/*
  We want a golden-section search the first iteration.  Force this by setting
  movement equal to zero.
*/

   movement = trial = 0.0 ;

/*
   Main loop.
*/

   for (iter=0 ; iter<itmax ; iter++) {

/*
   This test is more sophisticated than it looks.  It tests the closeness
   of xright and xleft (relative to small_dist), AND makes sure that x0 is
   near the midpoint of that interval.
*/

      small_step = fabs ( x0 ) ;
      if (small_step < 1.0)
         small_step = 1.0 ;
      small_step *= tol ;
      small_dist = 2.0 * small_step ;

      xmid = 0.5 * (xleft + xright) ;

      if (fabs ( x0 - xmid )  <=  (small_dist - 0.5 * (xright - xleft)))
         break ;

/*
   Avoid refining function to limits of precision
*/

      if ((iter >= 4)  &&  ((fabs(y2 - y0) / (fabs(y0) + 1.0)) < eps))
         break ;

      if (fabs ( movement )  >  small_step) {  // Try parabolic only if moving
#if DEBUG
         printf ( "\nTrying parabolic:" ) ;
#endif
         temp1 = (x0 - x2) * (y0 - y1) ;
         temp2 = (x0 - x1) * (y0 - y2) ;
         numer = (x0 - x1) * temp2 - (x0 - x2) * temp1 ;
         denom = 2. * (temp1 - temp2) ;
         testdist = movement ;     // Intervals must get smaller
         movement = trial ;
         if (fabs(denom) > 1.e-40)
            trial = numer / denom ; // Parabolic estimate of minimum
         else 
            trial = 1.e40 ;

         temp1 = trial + x0 ;
         if ((2.0 * fabs ( trial ) < fabs ( testdist ))// If shrinking
          && (temp1 > xleft) && (temp1 < xright)) {    // And safely in bounds
            this_x = temp1 ;                           // Use parabolic estimate
            if ((this_x - xleft  <  small_dist)  ||    // Cannot get too close
                (xright - this_x <  small_dist))       // to the endpoints
               trial = (x0 < xmid)  ?  small_step  :  -small_step ;
#if DEBUG
            printf ( " GOOD" ) ;
#endif
            }
         else {  // Punt via golden section because cannot use parabolic
            movement = (xmid > x0)  ?  xright - x0  :  xleft - x0 ;
            trial = .3819660 * movement ;
#if DEBUG
            printf ( " POOR" ) ;
#endif
            }
         }
      else { // Must use golden section due to insufficient movement
#if DEBUG
         printf ( "\nTrying golden." ) ;
#endif
         movement = (xmid > x0)  ?  xright - x0  :  xleft - x0 ;
         trial = .3819660 * movement ;
         }

      if (fabs (trial)  >=  small_step)     // Make sure we move a good distance
         this_x = x0 + trial ;
      else
         this_x = (trial > 0.0)  ?  x0 + small_step  :  x0 - small_step ;

/*
   Evaluate the function here.
*/

      this_y = c_func ( this_x ) ;
#if DEBUG
      printf ( " Eval err at %lf = %lf", this_x, this_y ) ;
#endif

/*
   Insert this new point in the correct position in the 'best' hierarchy
*/

      if (this_y >= y0) {    // Improvement
         if (this_x < x0)
            xright = x0 ;
         else
            xleft = x0 ;
         x2 = x1 ;
         x1 = x0 ;
         x0 = this_x ;
         y2 = y1 ;
         y1 = y0 ;
         y0 = this_y ;
         }

      else {                  // No improvement
         if (this_x >= x0)
            xright = this_x ;
         else
            xleft = this_x ;

         if ((this_y >= y1)  ||  (x1 == x0)) {
            x2 = x1 ;
            x1 = this_x ;
            y2 = y1 ;
            y1 = this_y ;
            }
         else if ((this_y >= y2)  ||  (x2 == x0)  ||  (x2 == x1)) {
            x2 = this_x ;
            y2 = this_y ;
            }
         }
      }
 
   *xa = xleft ;
   *xb = x0 ;
   *xc = xright ;

   return y0 ;
}