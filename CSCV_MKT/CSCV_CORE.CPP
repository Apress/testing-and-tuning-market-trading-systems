/******************************************************************************/
/*                                                                            */
/*  CSCVCORE - Combinatorially symmetric cross validation core routine        */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>

double criter ( int n , double *returns ) ;

double cscvcore (
   int ncases ,         // Number of columns in returns matrix (change fastest)
   int n_systems ,      // Number of rows (competitors); should be large enough to reduce granularity
   int n_blocks ,       // Number of blocks (even!) into which the cases will be partitioned
   double *returns ,    // N_systems by ncases matrix of returns, case changing fastest
   int *indices ,       // Work vector n_blocks long
   int *lengths ,       // Work vector n_blocks long
   int *flags ,         // Work vector n_blocks long
   double *work ,       // Work vector ncases long
   double *is_crits ,   // Work vector n_systems long
   double *oos_crits    // Work vector n_systems long
   /* double *logit */  // Work vector Comb(S,S/2) long (Optional, if you want logits)
   )
{
   int i, ic, isys, ibest, n, ncombo, iradix, istart, nless ;
   double best, rel_rank ;

/*
   Find the starting index and length of each of the n_blocks submatrices.
   Ideally, ncases should be an integer multiple of n_blocks so that
   all submatrices are the same size.
*/

   n_blocks = n_blocks / 2 * 2 ;   // Make sure it's even
   istart = 0 ;
   for (i=0 ; i<n_blocks ; i++) {
      indices[i] = istart ;        // Block starts here
      lengths[i] = (ncases - istart) / (n_blocks-i) ; // It contains this many cases
      istart += lengths[i] ;       // Next block
      }

/*
   Initialize
*/

   nless = 0 ;   // Will count the number of time OOS of best <= median OOS, for prob

   for (i=0 ; i<n_blocks / 2 ; i++)   // Identify the training set blocks
      flags[i] = 1 ;

   for ( ; i<n_blocks ; i++)          // And the test set blocks
      flags[i] = 0 ;

/*
   Main loop processes all combinations of blocks   
*/
   
   for (ncombo=0; ; ncombo++) {

/*
   Compute training-set (IS) criterion for each candidate system
*/

      for (isys=0 ; isys<n_systems ; isys++) { // Each row of returns matrix
         n = 0 ;                               // Counts cases in training set
         for (ic=0 ; ic<n_blocks ; ic++) {     // For all blocks (sub-matrices)
            if (flags[ic]) {                   // If this block is in the training set
               for (i=indices[ic] ; i<indices[ic]+lengths[ic] ; i++) // For every case in this block
                  work[n++] = returns[isys*ncases+i] ;
               }
            }

         is_crits[isys] = criter ( n , work ) ;
         }


/*
   Compute (OOS) criterion for each candidate system
*/

      for (isys=0 ; isys<n_systems ; isys++) { // Each column of returns matrix
         n = 0 ;                               // Counts cases in OOS set
         for (ic=0 ; ic<n_blocks ; ic++) {     // For all blocks (sub-matrices)
            if (! flags[ic]) {                 // If this block is in the OOS set
               for (i=indices[ic] ; i<indices[ic]+lengths[ic] ; i++) // For every case in this block
                  work[n++] = returns[isys*ncases+i] ;
               }
            }

         oos_crits[isys] = criter ( n , work ) ;
         }

/*
   Determine the relative rank within OOS of the system which had best IS performance.
   Optionally convert this to a logit.
*/

      for (isys=0 ; isys<n_systems ; isys++) {  // Find the best system IS
         if (isys == 0  ||  is_crits[isys] > best) {
            best = is_crits[isys] ;
            ibest = isys ;
            }
         }

      best = oos_crits[ibest] ;  // This is the OOS value for the best system IS
      n = 0 ;
      for (isys=0 ; isys<n_systems ; isys++) {
         if (isys == ibest  ||  best >= oos_crits[isys]) // Insurance against fpt error
            ++n ;
         }

      rel_rank = (double) n / (n_systems + 1) ;
      // logit[ncombo] = log ( rel_rank / (1.0 - rel_rank) ) ;  // Optional
      // See the original paper for interesting uses for the logit.

      if (rel_rank <= 0.5)   // Is the IS best at or below the OOS median?
         ++nless ;

/*
   Move to the next combination
*/

      n = 0 ;
      for (iradix=0 ; iradix<n_blocks-1 ; iradix++) {
         if (flags[iradix] == 1) {
            ++n ;                          // This many flags up to and including this one at iradix
            if (flags[iradix+1] == 0) {
               flags[iradix] = 0 ;
               flags[iradix+1] = 1 ;
               for (i=0 ; i<iradix ; i++) {  // Must reset everything below this change point
                  if (--n > 0)
                     flags[i] = 1 ;
                  else
                     flags[i] = 0 ;
                  } // Filling in below
               break ;
               } // If next flag is 0
            } // If this flag is 1
         } // For iradix

      if (iradix == n_blocks-1) {
         ++ncombo ;   // Must count this last one
         break ;
         }
      } // Main loop processes all combinations

   return (double) nless / ncombo ;
}
