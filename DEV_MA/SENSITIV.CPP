/******************************************************************************/
/*                                                                            */
/*   SENSITIV.CPP - Compute and optionally print parameter sensitivity curves */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "headers.h"

int sensitivity (
   double (*criter) ( double * , int ) , // Crit function maximized
   int nvars ,           // Number of variables
   int nints ,           // Number of first variables that are integers
   int npoints ,         // Number of points at which to evaluate performance
   int nres ,            // Number of resolved points across plot
   int mintrades ,       // Minimum number of trades
   double *best ,        // Optimal parameters
   double *low_bounds ,  // Lower bounds for parameters
   double *high_bounds   // And upper
   )
{
   int i, ivar, ipoint, ival, k ;
   double label_frac, hist_frac, rval, *vals, *params, maxval ;
   FILE *fp ;

/*
   Allocate memory and open the log file for writing
*/

   vals = (double *) malloc ( npoints * sizeof(double) ) ;
   params = (double *) malloc ( nvars * sizeof(double) ) ;
   if (vals == NULL  || params == NULL) {
      if (vals != NULL)
         free ( vals ) ;
      if (params != NULL)
         free ( params ) ;
      return 1 ;
      }

   if (fopen_s ( &fp , "SENS.LOG" , "wt" )) {
      free ( vals ) ;
      free ( params ) ;
      return 1 ;
      }


   for (ivar=0 ; ivar<nvars ; ivar++) {
      for (i=0 ; i<nvars ; i++)
         params[i] = best[i] ;

/*
   Process integer parameters
*/

      if (ivar < nints) {

         fprintf ( fp , "\n\nSensitivity curve for integer parameter %d (optimum=%d)\n", ivar+1, (int) (best[ivar] + 1.e-10) ) ;

         label_frac = (high_bounds[ivar] - low_bounds[ivar] + 0.99999999) / (npoints - 1) ;
         for (ipoint=0 ; ipoint<npoints ; ipoint++) {
            ival = (int) (low_bounds[ivar] + ipoint * label_frac) ;
            params[ivar] = ival ;
            vals[ipoint] = criter ( params , mintrades ) ;
            if (ipoint == 0  ||  vals[ipoint] > maxval)
               maxval = vals[ipoint] ;
            }

         hist_frac = (nres + 0.9999999) / maxval ;
         for (ipoint=0 ; ipoint<npoints ; ipoint++) {
            ival = (int) (low_bounds[ivar] + ipoint * label_frac) ;
            fprintf ( fp , "\n%6d|", ival ) ;
            k = (int) (vals[ipoint] * hist_frac) ;
            for (i=0 ; i<k ; i++)
               fprintf ( fp , "*" ) ;
            }
         }

/*
   Process real parameters
*/

      else {

         fprintf ( fp , "\n\nSensitivity curve for real parameter %d (optimum=%.4lf)\n", ivar+1, best[ivar] ) ;

         label_frac = (high_bounds[ivar] - low_bounds[ivar]) / (npoints - 1) ;
         for (ipoint=0 ; ipoint<npoints ; ipoint++) {
            rval = low_bounds[ivar] + ipoint * label_frac ;
            params[ivar] = rval ;
            vals[ipoint] = criter ( params , mintrades ) ;
            if (ipoint == 0  ||  vals[ipoint] > maxval)
               maxval = vals[ipoint] ;
            }

         hist_frac = (nres + 0.9999999) / maxval ;
         for (ipoint=0 ; ipoint<npoints ; ipoint++) {
            rval = low_bounds[ivar] + ipoint * label_frac ;
            fprintf ( fp , "\n%10.3lf|", rval ) ;
            k = (int) (vals[ipoint] * hist_frac) ;
            for (i=0 ; i<k ; i++)
               fprintf ( fp , "*" ) ;
            }
         }
      }
            
   fclose ( fp ) ;
   free ( vals ) ;
   free ( params ) ;
   return 0 ;
}