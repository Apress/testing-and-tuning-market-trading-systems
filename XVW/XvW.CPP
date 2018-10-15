/******************************************************************************/
/*                                                                            */
/*  XvW - Explore optimistic bias of cross validation over walkforward        */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>


/*
--------------------------------------------------------------------------------

   Normal CDF   Accurate to 7.5 e-8

--------------------------------------------------------------------------------
*/

double normal_cdf ( double z )
{
   double zz = fabs ( z ) ;
   double pdf = exp ( -0.5 * zz * zz ) / sqrt ( 2.0 * 3.141592653589793 ) ;
   double t = 1.0 / (1.0 + zz * 0.2316419) ;
   double poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t -
                     0.356563782) * t + 0.319381530) * t ;
   return (z > 0.0)  ?  1.0 - pdf * poly  :  pdf * poly ;
}


/*
--------------------------------------------------------------------------------

   Quicksort

--------------------------------------------------------------------------------
*/

void qsortd ( int first , int last , double *data )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortd ( first , upper , data ) ;
   if (lower < last)
      qsortd ( lower , last , data ) ;
}


/*
--------------------------------------------------------------------------------

   This is a random int generator suggested by Marsaglia in his DIEHARD suite.
   It provides a great combination of speed and quality.

   We also have unifrand(), a random 0-1 generator.

--------------------------------------------------------------------------------
*/

static unsigned int Q[256], carry=362436 ;
static int MWC256_initialized = 0 ;
static int MWC256_seed = 123456789 ;

void RAND32M_seed ( int iseed ) { // Optionally set seed
   MWC256_seed = iseed ;
   MWC256_initialized = 0 ;
   }

unsigned int RAND32M ()
{
   unsigned _int64 t ;
   unsigned _int64 a=809430660 ;
   static unsigned char i=255 ;

   if (! MWC256_initialized) {
      unsigned int k,j=MWC256_seed ;
      MWC256_initialized = 1 ;
      for (k=0 ; k<256 ; k++) {
         j = 69069 * j + 12345 ; // This overflows, doing an automatic mod 2^32
         Q[k] = j ;
         }
      }

   t = a * Q[++i] + carry ;  // This is the 64-bit op, forced by a being 64-bit
   carry = (unsigned int) (t >> 32) ;
   Q[i] = (unsigned int) (t & 0xFFFFFFFF) ;
   return Q[i] ;
}


double unifrand ()
{
   double mult = 1.0 / 0xFFFFFFFF ;
   return mult * RAND32M() ;
}


/*
--------------------------------------------------------------------------------

   Local routine computes a single indicator, the linear slope of a price block,
   and a single target, the price change over a specified lookahead

--------------------------------------------------------------------------------
*/

void ind_targ (
   int lookback ,    // Window length for computing slope indicator
   int lookahead ,   // Window length for computing target
   double *x ,       // Pointer to current price
   double *ind ,     // Returns indicator value (linear slope across lookback)
   double *targ      // Returns target value (price change over lookahead)
   )
{
   int i ;
   double *pptr, coef, slope, denom ;

   pptr = x - lookback + 1 ;     // Indicator lookback window starts here
   slope = 0.0 ;                 // Will sum slope here
   denom = 0.0 ;                 // Will sum normalizer here

   for (i=0 ; i<lookback ; i++) {
      coef = 2.0 * i / (lookback - 1.0) - 1.0 ;
      denom += coef * coef ;
      slope += coef * *pptr++ ;
      }

   *ind = slope / denom ;
   *targ = x[lookahead] - x[0] ;
}


/*
--------------------------------------------------------------------------------

   Local routine computes beta coefficient for simple linear regression

--------------------------------------------------------------------------------
*/

void find_beta (
   int ntrn ,         // Number of cases in data matrix (which has 2 columns)
   double *data ,     // ntrn by 2 data matrix of indicators and target
   double *beta ,     // Beta coefficient
   double *constant   // Constant
   )
{
   int i ;
   double *dptr, x, y, xmean, ymean, xy, xx ;

   xmean = ymean = xy = xx = 0.0 ;
   dptr = data ;

   for (i=0 ; i<ntrn ; i++) {
      xmean += *dptr++ ;
      ymean += *dptr++ ;
      }

   xmean /= ntrn ;
   ymean /= ntrn ;

   dptr = data ;
   for (i=0 ; i<ntrn ; i++) {
      x = *dptr++ - xmean ;
      y = *dptr++ - ymean ;
      xy += x * y ;
      xx += x * x ;
      }

   *beta = xy / (xx + 1.e-60) ;
   *constant = ymean - *beta * xmean ;
}


/*
--------------------------------------------------------------------------------

   Main routine

--------------------------------------------------------------------------------
*/

int main (
   int argc ,    // Number of command line arguments (includes prog name)
   char *argv[]  // Arguments (prog name is argv[0])
   )
{
   int i, j, ncases, ncols, nprices, lookback, lookahead, ntrain, ntest, nfolds, omit ;
   int itest, nt, n_OOS_X, n_OOS_W, irep, nreps, ncases_save ;
   int istart, istop, n_done, ifold, n_in_fold, seed ;
   double dtemp, *x, *data, *data_save, *trn_ptr, *test_ptr, beta, constant, pred, *dptr, *optr ;
   double *OOS, OOS_mean_X, OOS_mean_W, mean_W, mean_X, ss_W, ss_X, denom, t, trend, save_trend ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 11) {
      printf ( "\nUsage: XVW  nprices  trend  lookback  lookahead  ntrain  ntest  nfolds  omit  nreps  seed" ) ;
      printf ( "\n  nprices - Total number of prices (bars in history)" ) ;
      printf ( "\n  trend - Amount of trending, 0 for pure random walk" ) ;
      printf ( "\n  lookback - historical window length for indicator" ) ;
      printf ( "\n  lookahead - Bars into future for target" ) ;
      printf ( "\n  ntrain - Number of cases in walkforward training set" ) ;
      printf ( "\n  ntest - Number of cases in walkforward test set" ) ;
      printf ( "\n  nfolds - Number of XVAL folds" ) ;
      printf ( "\n  omit - Omit this many cases from end of training window" ) ;
      printf ( "\n  nreps - Number of replications" ) ;
      printf ( "\n  seed - Random seed" ) ;
      exit ( 1 ) ;
      }

   nprices = atoi ( argv[1] ) ;
   trend = atof ( argv[2] ) ;
   lookback = atoi ( argv[3] ) ;
   lookahead = atoi ( argv[4] ) ;
   ntrain = atoi ( argv[5] ) ;
   ntest = atoi ( argv[6] ) ;
   nfolds = atoi ( argv[7] ) ;
   omit = atoi ( argv[8] ) ;
   nreps = atoi ( argv[9] ) ;
   seed = atoi ( argv[10] ) ;
#else
   nprices = 1000 ;
   trend = 0.0 ;
   lookback = 25 ;
   lookahead = 5 ;
   ntrain = 100 ;
   ntest = 50 ;
   nfolds = 900 ;
   omit = 4 ;
   nreps = 100 ;
   seed = 411 ;
#endif

      if (nprices < 2  ||  lookback < 2  ||  lookahead < 1  ||
          ntrain < 2  ||  ntest < 1  ||  nfolds < 2  ||  omit < 0) {
      if (nprices < lookback + lookahead + ntrain + ntest + 10)
         printf ( "\nNprices must be at least lookback + lookahead + ntrain + ntest + 10" ) ;
      printf ( "\nUsage: XVW  nprices  trend  lookback  lookahead  ntrain  ntest  nfolds  omit  nreps  seed" ) ;
      printf ( "\n  nprices - Total number of prices (bars in history)" ) ;
      printf ( "\n  trend - Amount of trending, 0 for pure random walk" ) ;
      printf ( "\n  lookback - historical window length for indicator" ) ;
      printf ( "\n  lookahead - Bars into future for target" ) ;
      printf ( "\n  ntrain - Number of cases in training set" ) ;
      printf ( "\n  ntest - Number of cases in test set" ) ;
      printf ( "\n  nfolds - Number of XVAL folds" ) ;
      printf ( "\n  omit - Omit this many cases from end of training window" ) ;
      printf ( "\n  nreps - Number of replications" ) ;
      printf ( "\n  seed - Random seed" ) ;
      exit ( 1 ) ;
      }

   printf ( "\n\nnprices=%d  trend=%.3lf  lookback=%d  lookahead=%d  ntrain=%d  ntest=%d  nfolds=%d  omit=%d  nreps=%d  seed=%d",
            nprices, trend, lookback, lookahead, ntrain, ntest, nfolds, omit, nreps, seed ) ;

/*
   Initialize
*/

   save_trend = trend ;
   ncols = 2 ;   // Hard-programmed into this demonstration (1 predictor + target)
   x = (double *) malloc ( nprices * sizeof(double) ) ;
   data = (double *) malloc ( ncols * nprices * sizeof(double) ) ;  // More than we need, but simple
   if (omit)
      data_save = (double *) malloc ( ncols * nprices * sizeof(double) ) ;  // More than we need, but simple
   OOS = (double *) malloc ( nprices * sizeof(double) ) ;       // Ditto

   mean_W = mean_X = ss_W = ss_X = 0.0 ;

   RAND32M_seed ( seed ) ;

/*
   This simply replicates the test many times in one run to get p-values
   It is not a MCPT.
*/

   for (irep=0 ; irep<nreps ; irep++) {
      printf ( "\n\n%.2lf %%", 100.0 * irep / nreps ) ;

/*
   Generate the log prices as a random walk,
   and then compute the dataset, which is a 2-column matrix.
   The first column is the indicator and the second column is the corresponding target.
*/

      trend = save_trend ;
      x[0] = 0.0 ;
      for (i=1 ; i<nprices ; i++) {
         if ((i+1) % 50 == 0)   // Reverse the trend every 50 days
            trend = -trend ;
         x[i] = x[i-1] + trend + unifrand() + unifrand() - unifrand() - unifrand() ;
         }

      ncases = 0 ;
      for (i=lookback-1 ; i<nprices-lookahead ; i++) {
         ind_targ ( lookback , lookahead , x+i , data+ncols*ncases , data+ncols*ncases+1 ) ;
         ++ncases ;
         }

   if (omit)   // XVAL folds get fancy if buffers
      memcpy ( data_save , data , ncases * ncols * sizeof(double) ) ;

/*
   The number of folds cannot exceed the number of cases
*/

   if (nfolds > ncases) {
      printf ( "\n\nNumber of XVAL folds reduced from %d to %d.  Press any key to continue...", nfolds, ncases ) ;
      nfolds = ncases ;
      _getch () ;
      }


/*
-------------------------------------
   Compute the walkforward OOS values
-------------------------------------
*/

      trn_ptr = data ;            // Point to training set
      istart = ntrain ;           // First OOS case
      n_OOS_W = 0 ;               // Counts OOS cases

      for (ifold=0 ;; ifold++) {
         test_ptr = trn_ptr + ncols * ntrain ;    // Test set starts right after training set
         if (test_ptr >= data + ncols * ncases )  // No test cases left?
            break ;
         find_beta ( ntrain - omit , trn_ptr , &beta , &constant ) ; // Training phase
         nt = ntest ;
         if (nt > ncases - istart)                  // Last fold may be incomplete
            nt = ncases - istart ;
         for (itest=0 ; itest<nt ; itest++) {       // For every case in the test set
            assert ( test_ptr + 1 < data + ncols * ncases ) ; // Verify testing valid data
            pred = beta * *test_ptr++ + constant ;  // test_ptr points to target after this line of code
            if (pred > 0.0)
               OOS[n_OOS_W++] = *test_ptr ;
            else
               OOS[n_OOS_W++] = - *test_ptr ;
            ++test_ptr ;    // Advance to indicator for next test case
            }
         istart += nt ;            // First OOS case for next fold
         trn_ptr += ncols * nt ;   // Advance training set to next fold
         }

/*
   Analyze the walkforward OOS results
*/

      OOS_mean_W = 0.0 ;
      for (i=0 ; i<n_OOS_W ; i++)
         OOS_mean_W += OOS[i] ;

      OOS_mean_W /= n_OOS_W ;

      printf ( "\nWALK n OOS = %d  Mean = %.4lf", n_OOS_W, OOS_mean_W ) ;

/*
------------------------------------------------------
   Compute the XVAL OOS values.  For each fold...

   if omit
      copy istart::istop to end of data (OOS)
      if first fold
         copy istop+omit::ncases to 0
         ncases -= n_in_fold + omit
      else if last fold
         copy 0::istart-omit to 0
         ncases -= n_in_fold + omit
      else central fold
         copy 0::istart-omit to 0
         copy istop+omit::ncases to istart-omit
         ncases -= n_in_fold + 2 * omit
   else
      if prior to last fold
         swap OOS to end
      ncases -= n_in_fold

   Train
   restore ncases
   Test

   if (not omit AND not last fold)
      swap OOS back from end

------------------------------------------------------
*/

      istart = 0 ;         // OOS start = training data start
      n_done = 0 ;         // Number of cases treated as OOS so far
      n_OOS_X = 0 ;        // Counts OOS cases
      ncases_save = ncases ;

      for (ifold=0 ; ifold<nfolds ; ifold++) {

         n_in_fold = (ncases - n_done) / (nfolds - ifold) ;
         istop = istart + n_in_fold ;  // One past OOS stop

         if (omit) {
            memcpy ( data+(ncases-n_in_fold)*ncols , data_save+istart*ncols , n_in_fold*ncols*sizeof(double) ) ;

            if (ifold == 0) {   // First (leftmost) fold
               memcpy ( data , data_save+(istop+omit)*ncols , (ncases-istop-omit)*ncols*sizeof(double) ) ;
               ncases -= n_in_fold + omit ;
               }
            else if (ifold == nfolds-1) {  // Last (rightmost) fold
               memcpy ( data , data_save , (istart-omit)*ncols*sizeof(double) ) ;
               ncases -= n_in_fold + omit ;
               }
            else {                    // Interior fold
               ncases = 0 ;
               if (istart > omit) {   // We have at least one case prior to OOS block
                  memcpy ( data , data_save , (istart-omit)*ncols*sizeof(double) ) ;
                  ncases = istart - omit ;
                  }
               if (ncases_save > istop+omit) {  // We have at least one case after OOS block
                  memcpy ( data+ncases*ncols , data_save+(istop+omit)*ncols ,
                          (ncases_save-istop-omit)*ncols*sizeof(double) ) ;
                  ncases += ncases_save - istop - omit ;
                  }
               }
            }

         else {
            // Swap this OOS set to end of dataset if it's not already there
            if (ifold < nfolds-1) {
               for (i=istart ; i<istop ; i++) {
                  dptr = data + i * ncols ;                            // Swap from here
                  optr = data + (ncases-n_in_fold+i-istart) * ncols ;  // To here
                  for (j=0 ; j<ncols ; j++) {
                     dtemp = dptr[j] ;
                     dptr[j] = optr[j] ;
                     optr[j] = dtemp ;
                     }
                  } // For all OOS cases, swapping
               } // If prior to last fold
            else
               assert ( ncases-n_in_fold-istart == 0 ) ;
            ncases -= n_in_fold ;
            } // Else not omit

/*
   Train and test this XVAL fold
   When we prepared to process this fold, we reduced ncases to remove the OOS set and any omitted buffer.
   As soon as we finish training, we restore it back to its full value.
*/

         find_beta ( ncases , data , &beta , &constant ) ;  // Training phase
         ncases = ncases_save ; // Was reduced for training but now done training

         test_ptr = data+(ncases-n_in_fold)*ncols ;   // OOS test set starts right after training set
         for (itest=0 ; itest<n_in_fold ; itest++) {  // For every case in the test set
            pred = beta * *test_ptr++ + constant ;    // test_ptr points to target after this line of code
            if (pred > 0.0)
               OOS[n_OOS_X++] = *test_ptr ;
            else
               OOS[n_OOS_X++] = - *test_ptr ;
            ++test_ptr ;    // Advance to indicator for next test case
            }

/*
   Swap this OOS set back from end of dataset if it was swapped there
*/

         if (omit == 0  &&  ifold < nfolds-1) {
            for (i=istart ; i<istop ; i++) {
               dptr = data + i * ncols ;
               optr = data + (ncases-n_in_fold+i-istart) * ncols ;
               for (j=0 ; j<ncols ; j++) {
                  dtemp = dptr[j] ;
                  dptr[j] = optr[j] ;
                  optr[j] = dtemp ;
                  }
               }
            }

         istart = istop ;       // Advance the OOS set
         n_done += n_in_fold ;  // Count the OOS cases we've done
         } // For ifold


/*
   Analyze the XVAL OOS results
*/

      OOS_mean_X = 0.0 ;
      for (i=0 ; i<n_OOS_X ; i++)
         OOS_mean_X += OOS[i] ;

      OOS_mean_X /= n_OOS_X ;

      printf ( "\nXVAL n OOS = %d  Mean = %.4lf", n_OOS_X, OOS_mean_X ) ;

/*
   We finished this replication of WALK and XVAL.
   Cumulate statistics for t-test
*/

      mean_W += OOS_mean_W ;
      mean_X += OOS_mean_X ;
      ss_W += OOS_mean_W * OOS_mean_W ;
      ss_X += OOS_mean_X * OOS_mean_X ;
      }  // For all replications

/*
   All replications are finished.  Do final computation and print results.
*/

   mean_W /= nreps ;
   mean_X /= nreps ;
   denom = ss_W + ss_X - nreps * (mean_W * mean_W + mean_X * mean_X) ;
   denom /= (double) nreps * (nreps - 1.0) ;
   denom = sqrt ( denom ) ;
   t = (mean_X - mean_W) / denom ;   // Two-sample t-test with unequal variances

   printf ( "\n\nnprices=%d  trend=%.3lf  lookback=%d  lookahead=%d  ntrain=%d  ntest=%d  nfolds=%d  omit=%d  nreps=%d  seed=%d",
            nprices, save_trend, lookback, lookahead, ntrain, ntest, nfolds, omit, nreps, seed ) ;

   printf ( "\n\nGrand XVAL = %.5lf (t=%.3lf)  WALK = %.5lf (t=%.3lf)  StdDev = %.5lf  t = %.3lf  rtail = %.5lf",
            mean_X, sqrt ( (double) nreps) * mean_X / sqrt ( ss_X/nreps - mean_X * mean_X ),
            mean_W, sqrt ( (double) nreps) * mean_W / sqrt ( ss_W/nreps - mean_W * mean_W ),
            denom, t, 1.0 - normal_cdf ( t ) ) ;
   _getch () ;  // Wait for user to press a key

   free ( x ) ;
   free ( data ) ;
   if (omit)
      free ( data_save ) ;
   free ( OOS ) ;

   return 0 ;
}
