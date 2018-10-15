/******************************************************************************/
/*                                                                            */
/*  CRITER - Criterion function for CSCV                                      */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>

#if 1
double criter ( int n , double *returns )
{
   int i ;
   double sum ;

   sum = 0.0 ;
   for (i=0 ; i<n ; i++)
      sum += returns[i] ;

   return sum / n ;
}

#else

double criter ( int n , double *returns )
{
   int i ;
   double win_sum, lose_sum ;

   win_sum = lose_sum = 1.e-60 ;

   for (i=0 ; i<n ; i++) {
      if (returns[i] > 0.0)
         win_sum += returns[i] ;
      else
         lose_sum -= returns[i] ;
      }

   return win_sum / lose_sum ;
}
#endif
