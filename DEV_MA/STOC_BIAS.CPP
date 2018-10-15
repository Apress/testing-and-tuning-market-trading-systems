/******************************************************************************/
/*                                                                            */
/*  STOC_BIAS - Roughly estimate training bias from data collected from       */
/*              a stochastic training procedure.                              */
/*                                                                            */
/******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <malloc.h>
#include "headers.h"

/*
   Constructor
*/

StocBias::StocBias (
   int nc
   )
{
   nreturns = nc ;
   ok = 1 ;
   collecting = 0 ;
   got_first_case = 0 ;


   IS_best = (double *) malloc ( nreturns * sizeof(double) ) ;
   OOS = (double *) malloc ( nreturns * sizeof(double) ) ;
   returns = (double *) malloc ( nreturns * sizeof(double) ) ;

   if (IS_best == NULL  ||  OOS == NULL  ||  returns == NULL) {
      if (IS_best != NULL) {
         free ( IS_best ) ;
         IS_best = NULL ;
         }
      if (OOS != NULL) {
         free ( OOS ) ;
         OOS = NULL ;
         }
      if (returns != NULL) {
         free ( returns ) ;
         returns = NULL ;
         }
      ok = 0 ;
      return ;
      }
}


/*
   Destructor
*/

StocBias::~StocBias ()
{
   if (IS_best != NULL)
      free ( IS_best ) ;
   if (OOS != NULL)
      free ( OOS ) ;
   if (returns != NULL)
      free ( returns ) ;
}


/*
   Called from stochastic optimizer to turn collecting on and off
   We must collect only during random or exhaustive search, never
   during an intelligent, guided search.
*/

void StocBias::collect ( int collect_data )
{
   collecting = collect_data ;
}


/*
   Process the current set of returns
*/

void StocBias::process ()
{
   int i ;
   double total , this_x ;

   if (! collecting)
      return ;

   total = 0.0 ;
   for (i=0 ; i<nreturns ; i++)
      total += returns[i] ;

   // Initialize if this is the first call

   if (! got_first_case) {
      got_first_case = 1 ;
      for (i=0 ; i<nreturns ; i++) {
         this_x = returns[i] ;
         IS_best[i] = total - this_x ;
         OOS[i] = this_x ;
         }
      }

   // Keep track of best if this is a subsequent call

   else {
      for (i=0 ; i<nreturns ; i++) {
         this_x = returns[i] ;
         if (total - this_x > IS_best[i]) {
            IS_best[i] = total - this_x ;
            OOS[i] = this_x ;
            }
         }
      }
}


/*
   This may be overkill, but it lets us keep 'returns' private.
   This is called by the criterion routine, and it tells that
   routine where to put the bar returns.
   We could have just as well made 'returns' public!
*/

double *StocBias::expose_returns ()
{
   return returns ;
}


/*
   Do the final computation after all cases have been processed.
   The normal situation will be for the supplied returns to be log bar returns.
   This works on the basis of total log return.
*/

void StocBias::compute (
   double *IS_return ,
   double *OOS_return ,
   double *bias
   )
{
   int i ;

   *IS_return = *OOS_return = 0.0 ;

   for (i=0 ; i<nreturns ; i++) {
      *IS_return += IS_best[i] ;
      *OOS_return += OOS[i] ;
      }

   *IS_return /= (nreturns - 1) ;     // Each IS_best is the sum of nreturns-1 returns
   *bias = *IS_return - *OOS_return ;
}