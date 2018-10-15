/******************************************************************************/
/*                                                                            */
/*  Overlap - Explore the effect of unobvious IS/OOS overlap in walkforward   */
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
   int i, ncols, ncases, nprices, lookback, lookahead, ntrain, ntest, omit, extra ;
   int ifold, nt, istart, itest, n_OOS, irep, nreps, p1_count ;
   double *x, *data, *trn_ptr, *test_ptr, beta, constant, pred, *oos ;
   double oos_mean, oos_ss, t, *save_t, rtail ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 9) {
      printf ( "\nUsage: Overlap  nprices  lookback  lookahead  ntrain  ntest  omit  extra  nreps" ) ;
      printf ( "\n  nprices - Total number of prices (bars in history)" ) ;
      printf ( "\n  lookback - historical window length for indicator" ) ;
      printf ( "\n  lookahead - Bars into future for target" ) ;
      printf ( "\n  ntrain - Number of cases in training set" ) ;
      printf ( "\n  ntest - Number of cases in test set" ) ;
      printf ( "\n  omit - Omit this many cases from end of training window" ) ;
      printf ( "\n  extra - Extra (beyond ntest) bars jumped for next fold" ) ;
      printf ( "\n  nreps - Number of replications" ) ;
      exit ( 1 ) ;
      }

   nprices = atoi ( argv[1] ) ;
   lookback = atoi ( argv[2] ) ;
   lookahead = atoi ( argv[3] ) ;
   ntrain = atoi ( argv[4] ) ;
   ntest = atoi ( argv[5] ) ;
   omit = atoi ( argv[6] ) ;
   extra = atoi ( argv[7] ) ;
   nreps = atoi ( argv[8] ) ;
#else
   nprices = 100000 ;
   lookback = 1000 ;
   lookahead = 1 ;
   ntrain = 100 ;
   ntest = 1 ;
   omit = 0 ;
   extra = 0 ;
   nreps = 51 ;
#endif

   nreps = nreps / 2 * 2 + 1 ;  // Force it to be odd

   if (nprices < 2  ||  lookback < 2  ||  lookahead < 1  ||
       ntrain < 2  ||  ntest < 1  ||  omit < 0  ||  extra < 0) {
      if (nprices < lookback + lookahead + ntrain + ntest + 10)
         printf ( "\nNprices must be at least lookback + lookahead + ntrain + ntest + 10" ) ;
      printf ( "\nUsage: Overlap  nprices  lookback  lookahead  ntrain  ntest  omit  extra  nreps" ) ;
      printf ( "\n  nprices - Total number of prices (bars in history)" ) ;
      printf ( "\n  lookback - historical window length for indicator" ) ;
      printf ( "\n  lookahead - Bars into future for target" ) ;
      printf ( "\n  ntrain - Number of cases in training set" ) ;
      printf ( "\n  ntest - Number of cases in test set" ) ;
      printf ( "\n  omit - Omit this many cases from end of training window" ) ;
      printf ( "\n  extra - Extra (beyond ntest) bars jumped for next fold" ) ;
      printf ( "\n  nreps - Number of replications" ) ;
      exit ( 1 ) ;
      }

   printf ( "\n\nnprices=%d  lookback=%d  lookahead=%d  ntrain=%d  ntest=%d  omit=%d  extra=%d",
            nprices, lookback, lookahead, ntrain, ntest, omit, extra ) ;

/*
   Initialize
*/

   ncols = 2 ;   // Hard-programmed into this demonstration (1 predictor + target)
   x = (double *) malloc ( nprices * sizeof(double) ) ;
   data = (double *) malloc ( ncols * nprices * sizeof(double) ) ;  // More than we need, but simple
   oos = (double *) malloc ( nprices * sizeof(double) ) ;       // Ditto
   save_t = (double *) malloc ( nreps * sizeof(double) ) ;

   p1_count = 0 ;  // Will count reps with right tail p <= 0.1

/*
   This simply replicates the test a few times in one run to get median t and p<=0.1 count.
   It is not a MCPT.
*/

   for (irep=0 ; irep<nreps ; irep++) {

/*
   Generate the log prices as a random walk,
   and then compute the dataset, which is a 2-column matrix.
   The first column is the indicator and the second column is the corresponding target.
*/

      x[0] = 0.0 ;
      for (i=1 ; i<nprices ; i++)
         x[i] = x[i-1] + unifrand() + unifrand() - unifrand() - unifrand() ;

      ncases = 0 ;
      for (i=lookback-1 ; i<nprices-lookahead ; i++) {
         ind_targ ( lookback , lookahead , x+i , data+ncols*ncases , data+ncols*ncases+1 ) ;
         ++ncases ;
         }

/*
   Compute the walkforward OOS values
*/

      trn_ptr = data ;            // Point to training set
      istart = ntrain ;           // First OOS case
      n_OOS = 0 ;                 // Counts OOS cases

      for (ifold=0 ;; ifold++) {
         test_ptr = trn_ptr + ncols * ntrain ;    // Test set starts right after training set
         if (test_ptr >= data + ncols * ncases )  // No test cases left?
            break ;
         find_beta ( ntrain - omit , trn_ptr , &beta , &constant ) ;
         nt = ntest ;
         if (nt > ncases - istart)                  // Last fold may be incomplete
            nt = ncases - istart ;
         for (itest=0 ; itest<nt ; itest++) {       // For every case in the test set
            assert ( test_ptr + 1 < data + ncols * ncases ) ; // Verify testing valid data
            pred = beta * *test_ptr++ + constant ;  // test_ptr points to target after this line of code
            if (pred > 0.0)
               oos[n_OOS++] = *test_ptr ;
            else
               oos[n_OOS++] = - *test_ptr ;
            ++test_ptr ;    // Advance to indicator for next test case
            }
         istart += nt + extra ;          // First OOS case for next fold
         trn_ptr += ncols * (nt + extra) ;   // Advance to next fold
         }

/*
   Analyze results
*/

      oos_mean = oos_ss = 0.0 ;
      for (i=0 ; i<n_OOS ; i++) {
         oos_mean += oos[i] ;
         oos_ss += oos[i] * oos[i] ;
         }

      oos_mean /= n_OOS ;
      oos_ss /= n_OOS ;                // Formula in next line is usually dangerous
      oos_ss -= oos_mean * oos_mean ;  // But it is fine here because oos_mean is not large

      if (oos_ss < 1.e-20)
         oos_ss = 1.e-20 ;

      t = sqrt((double) n_OOS) * oos_mean / sqrt ( oos_ss ) ;  // Compute t-score
      rtail = 1.0 - normal_cdf ( t ) ;  // Normal CDF is close enough when n_OOS is not small
      printf ( "\nMean = %.4lf  StdDev = %.4lf  t = %.4lf  p = %.4lf", oos_mean, sqrt(oos_ss), t, rtail ) ;      
      save_t[irep] = t ;

      if (rtail <= 0.1)
         ++p1_count ;    // In correct walkforward, this should happen about 1 out of 10 times

      }  // For all replications

   qsortd ( 0 , nreps-1 , save_t ) ;
   printf ( "\nn OOS = %d  Median t = %.4lf  Fraction with p<= 0.1 = %.3lf",
            n_OOS, save_t[nreps/2], (double) p1_count / nreps ) ;
   _getch () ;  // Wait for user to press a key

   free ( x ) ;
   free ( data ) ;
   free ( oos ) ;
   free ( save_t ) ;

   return 0 ;
}
