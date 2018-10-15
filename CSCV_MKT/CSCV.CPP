/******************************************************************************/
/*                                                                            */
/*  CSCV - Computationally symmetric cross validation.                        */
/*                                                                            */
/*  The core routine is in CSCV_CORE.CPP which handles any app                */
/*  This is the wrapper for applying CSCV to moving-average crossover         */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>

double cscvcore (
   int ncases ,         // Number of rows in returns matrix
   int n_systems ,      // Number of columns (competitors); should be large enough to reduce granularity
   int n_blocks ,       // Number of blocks (even!) into which the cases will be partitioned
   double *returns ,    // Ncases by n_systems matrix of returns
   int *indices ,       // Work vector n_blocks long
   int *lengths ,       // Work vector n_blocks long
   int *flags ,         // Work vector n_blocks long
   double *work ,       // Work vector ncases long
   double *is_crits ,   // Work vector n_systems long
   double *oos_crits    // Work vector n_systems long
   ) ;


/*
--------------------------------------------------------------------------------

   This is a random int generator suggested by Marsaglia in his DIEHARD suite.
   It provides a great combination of speed and quality.

   We also have unifrand(), a random 0-1 generator based on it.

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
   static double mult = 1.0 / 0xFFFFFFFF ;
   return mult * RAND32M() ;
}


/*
--------------------------------------------------------------------------------

   Local routine computes one-bar returns for all short-term and
   long-term lookbacks of a primitive moving-average crossover system.
   The computed returns matrix has max_lookback * (max_lookback-1) / 2 rows
   and nprices-max_lookback columns, which change fastest.
   Note that this is the transpose of the matrix in the original paper.

--------------------------------------------------------------------------------
*/

void get_returns (
   int nprices ,      // Number of log prices in 'prices'
   double *prices ,   // Log prices
   int max_lookback , // Maximum lookback to use
   double *returns    // Computed matrix of returns
   )
{
   int i, j, ishort, ilong, iret ;
   double ret, long_mean, long_sum, short_mean, short_sum ;

   iret = 0 ;   // Will index computed returns

   for (ilong=2 ; ilong<=max_lookback ; ilong++) {  // Long-term lookback
      for (ishort=1 ; ishort<ilong ; ishort++) {    // Short-term lookback

         // We have a pair of lookbacks.  Compute short-term and long-term moving averages.
         // The index of the first legal bar in prices is max_lookback-1, because
         // we will need max_lookback cases (including the decision bar) 
         // in the longest long-term moving average.
         // We must stop one bar before the end of the price array because we need
         // the next price to compute the return from the decision.

         for (i=max_lookback-1 ; i<nprices-1 ; i++) { // Compute performance across history

            if (i == max_lookback-1) { // Find the short-term and long-term moving averages for the first valid case.
               short_sum = 0.0 ;                 // Cumulates short-term lookback sum
               for (j=i ; j>i-ishort ; j--)
                  short_sum += prices[j] ;
               long_sum = short_sum ;            // Cumulates long-term lookback sum
               while (j>i-ilong)
                  long_sum += prices[j--] ;
               }

            else {                               // Update the moving averages
               short_sum += prices[i] - prices[i-ishort] ;
               long_sum += prices[i] - prices[i-ilong] ;
               }

            short_mean = short_sum / ishort ;
            long_mean = long_sum / ilong ;

            // We now have the short-term and long-term moving averages ending at bar i
            // Save the return produced by this decision bar.

            if (short_mean > long_mean)       // Long position
               ret = prices[i+1] - prices[i] ;
            else if (short_mean < long_mean)  // Short position
               ret = prices[i] - prices[i+1] ;
            else
               ret = 0.0 ;

            returns[iret++] = ret ;
            } // For i (decision bar)

         } // For ishort, all short-term lookbacks
      } // For ilong, all long-term lookbacks

   assert ( iret == (max_lookback * (max_lookback-1) / 2 * (nprices - max_lookback)) ) ;
}


/*
--------------------------------------------------------------------------------

   Optional main routine for testing with random numbers

--------------------------------------------------------------------------------
*/

int main (
   int argc ,    // Number of command line arguments (includes prog name)
   char *argv[]  // Arguments (prog name is argv[0])
   )
{
   int i, nprices, n_blocks, max_lookback, n_systems, n_returns, iseed ;
   int *indices, *lengths, *flags ;
   double save_trend, trend, *prices, *returns, *work, *is_crits, *oos_crits, prob ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 6) {
      printf ( "\nUsage: CSCV  nprices  n_blocks  trend  max_lookback  seed" ) ;
      printf ( "\n  nprices - number of prices" ) ;
      printf ( "\n  n_blocks - number of blocks into which cases are partitioned" ) ;
      printf ( "\n  trend - Amount of trending, 0 for flat system" ) ;
      printf ( "\n  max_lookback - Maximum moving-average lookback" ) ;
      printf ( "\n  seed - Random seed, any positive integer" ) ;
      exit ( 1 ) ;
      }

   nprices = atoi ( argv[1] ) ;
   n_blocks = atoi ( argv[2] ) ;
   save_trend = atof ( argv[3] ) ;
   max_lookback = atoi ( argv[4] ) ;
   iseed = atoi ( argv[5] ) ;
#else
   nprices = 1000 ;
   n_blocks = 4 ;
   save_trend = 0.0 ;
   max_lookback = 10 ;
   iseed = 12345 ;
#endif

   n_returns = nprices - max_lookback ;
   n_systems = max_lookback * (max_lookback-1) / 2 ;

   if (nprices < 2  ||  n_blocks < 2  ||  max_lookback < 2  ||  n_returns < n_blocks) {
      printf ( "\nUsage: CSCV  nprices  n_blocks  trend  max_lookback  seed" ) ;
      printf ( "\n  nprices - number of prices" ) ;
      printf ( "\n  n_blocks - number of blocks into which cases are partitioned" ) ;
      printf ( "\n  trend - Amount of trending, 0 for flat system" ) ;
      printf ( "\n  max_lookback - Maximum moving-average lookback" ) ;
      printf ( "\n  seed - Random seed, any positive integer" ) ;
      exit ( 1 ) ;
      }

   printf ( "\n\nnprices=%d  n_blocks=%d  trend=%.3lf  max_lookback=%d  n_systems=%d  n_returns=%d",
            nprices, n_blocks, save_trend, max_lookback, n_systems, n_returns ) ;

/*
   Initialize
*/

   RAND32M_seed ( iseed ) ;

   prices = (double *) malloc ( nprices * sizeof(double) ) ;
   returns = (double *) malloc ( n_systems * n_returns * sizeof(double) ) ;
   indices = (int *) malloc ( n_blocks * sizeof(int) ) ;
   lengths = (int *) malloc ( n_blocks * sizeof(int) ) ;
   flags = (int *) malloc ( n_blocks * sizeof(int) ) ;
   work = (double *) malloc ( n_returns * sizeof(double) ) ;
   is_crits = (double *) malloc ( n_systems * sizeof(double) ) ;
   oos_crits = (double *) malloc ( n_systems * sizeof(double) ) ;

   // Generate the log prices
   trend = save_trend ;
   prices[0] = 0.0 ;
   for (i=1 ; i<nprices ; i++) {
      if (i % 100 == 0) // Reverse the trend
         trend = -trend ;
      prices[i] = prices[i-1] + trend + unifrand() + unifrand() - unifrand() - unifrand() ;
      }

   get_returns ( nprices , prices , max_lookback , returns ) ;

   prob = cscvcore ( n_returns , n_systems , n_blocks , returns , indices ,
                     lengths , flags , work , is_crits , oos_crits ) ;

   // Done.  Print results and clean up.
   printf ( "\nProb = %.4lf", prob ) ;
   _getch () ;  // Wait for user to press a key

   free ( prices ) ;
   free ( returns ) ;
   free ( indices ) ;
   free ( lengths ) ;
   free ( flags ) ;
   free ( work ) ;
   free ( is_crits ) ;
   free ( oos_crits ) ;

   return 0 ;
}
