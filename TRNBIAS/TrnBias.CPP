/******************************************************************************/
/*                                                                            */
/*  TrnBias - Explore training bias                                           */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>


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

   Local routine computes optimal short-term and long-term lookbacks
   for a primitive moving-average crossover system

--------------------------------------------------------------------------------
*/

double opt_params (
   int which ,       // 0=mean return; 1=profit factor; 2=Sharpe ratio
   int ncases ,      // Number of log prices in X
   double *x ,       // Log prices
   int *short_term , // Returns optimal short-term lookback
   int *long_term    // Returns optimal long-term lookback
   )
{
   int i, j, ishort, ilong, ibestshort, ibestlong ;
   double short_sum, long_sum, short_mean, long_mean, total_return, best_perf ;
   double ret, win_sum, lose_sum, sum_squares, sr ;

   best_perf = -1.e60 ;                          // Will be best performance across all trials
   for (ilong=2 ; ilong<200 ; ilong++) {         // Trial long-term lookback
      for (ishort=1 ; ishort<ilong ; ishort++) { // Trial short-term lookback

         // We have a pair of lookbacks to try.
         // Cumulate performance for all valid cases

         total_return = 0.0 ;                    // Cumulate total return for this trial
         win_sum = lose_sum = 1.e-60 ;           // Cumulates for profit factor
         sum_squares = 1.e-60 ;                  // Cumulates for Sharpe ratio

         for (i=ilong-1 ; i<ncases-1 ; i++) {    // Compute performance across history

            if (i == ilong-1) { // Find the short-term and long-term moving averages for the first valid case.
               short_sum = 0.0 ;                 // Cumulates short-term lookback sum
               for (j=i ; j>i-ishort ; j--)
                  short_sum += x[j] ;
               long_sum = short_sum ;            // Cumulates long-term lookback sum
               while (j>i-ilong)
                  long_sum += x[j--] ;
               }

            else {                               // Update the moving averages
               short_sum += x[i] - x[i-ishort] ;
               long_sum += x[i] - x[i-ilong] ;
               }

            short_mean = short_sum / ishort ;
            long_mean = long_sum / ilong ;

            // We now have the short-term and long-term moving averages ending at day i
            // Take our position and cumulate performance

            if (short_mean > long_mean)       // Long position
               ret = x[i+1] - x[i] ;
            else if (short_mean < long_mean)  // Short position
               ret = x[i] - x[i+1] ;
            else
               ret = 0.0 ;

            total_return += ret ;
            sum_squares += ret * ret ;
            if (ret > 0.0)
               win_sum += ret ;
            else
               lose_sum -= ret ;

            } // For i, summing performance for this trial

         // We now have the performance figures across the history
         // Keep track of the best lookbacks

         if (which == 0) {      // Mean return criterion
            total_return /= ncases - ilong ;
            if (total_return > best_perf) {
               best_perf = total_return ;
               ibestshort = ishort ;
               ibestlong = ilong ;
               }
            }

         else if (which == 1  &&  win_sum / lose_sum > best_perf) { // Profit factor criterion
            best_perf = win_sum / lose_sum ;
            ibestshort = ishort ;
            ibestlong = ilong ;
            }

         else if (which == 2) { // Sharpe ratio criterion
            total_return /= ncases - ilong ;   // Now mean return
            sum_squares /= ncases - ilong ;
            sum_squares -= total_return * total_return ;  // Variance (may be zero!)
            sr = total_return / (sqrt ( sum_squares ) + 1.e-8) ;
            if (sr > best_perf) { // Profit factor criterion
               best_perf = sr ;
               ibestshort = ishort ;
               ibestlong = ilong ;
               }
            }

         } // For ishort, all short-term lookbacks
      } // For ilong, all long-term lookbacks

   *short_term = ibestshort ;
   *long_term = ibestlong ;

   return best_perf ;
}


/*
--------------------------------------------------------------------------------

   Local routine tests a trained crossover system
   This computes the mean return.  Users may wish to change it to
   compute other criteria.

--------------------------------------------------------------------------------
*/

double test_system (
   int ncases ,
   double *x ,
   int short_term ,
   int long_term
   )
{
   int i, j ;
   double sum, short_mean, long_mean ;

   sum = 0.0 ;                          // Cumulate performance for this trial
   for (i=long_term-1 ; i<ncases-1 ; i++) { // Sum performance across history
      short_mean = 0.0 ;                // Cumulates short-term lookback sum
      for (j=i ; j>i-short_term ; j--)
         short_mean += x[j] ;
      long_mean = short_mean ;          // Cumulates short-term lookback sum
      while (j>i-long_term)
         long_mean += x[j--] ;
      short_mean /= short_term ;
      long_mean /= long_term ;
      // We now have the short-term and long-term means ending at day i
      // Take our position and cumulate return
      if (short_mean > long_mean)       // Long position
         sum += x[i+1] - x[i] ;
      else if (short_mean < long_mean)  // Short position
         sum -= x[i+1] - x[i] ;
      } // For i, summing performance for this trial
   sum /= ncases - long_term ;              // Mean return across the history we just tested

   return sum ;
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
   int i, which, ncases, irep, nreps, short_lookback, long_lookback ;
   double save_trend, trend, *x, IS_perf, OOS_perf, IS_mean, OOS_mean ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 5) {
      printf ( "\nUsage: TrnBias  which  ncases trend  nreps" ) ;
      printf ( "\n  which - 0=mean return  1=profit factor  2=Sharpe ratio" ) ;
      printf ( "\n  ncases - number of training and test cases" ) ;
      printf ( "\n  trend - Amount of trending, 0 for flat system" ) ;
      printf ( "\n  nreps - number of test replications" ) ;
      exit ( 1 ) ;
      }

   which = atoi ( argv[1] ) ;
   ncases = atoi ( argv[2] ) ;
   save_trend = atof ( argv[3] ) ;
   nreps = atoi ( argv[4] ) ;
#else
   which = 1 ;
   ncases = 1000 ;
   save_trend = 0.2 ;
   nreps = 10 ;
#endif

   if (ncases < 2  ||  which < 0  ||  which > 2  ||  nreps < 1) {
      printf ( "\nUsage: TrnBias  which  ncases trend  nreps" ) ;
      printf ( "\n  which - 0=mean return  1=profit factor  2=Sharpe ratio" ) ;
      printf ( "\n  ncases - number of training and test cases" ) ;
      printf ( "\n  trend - Amount of trending, 0 for flat system" ) ;
      printf ( "\n  nreps - number of test replications" ) ;
      exit ( 1 ) ;
      }

   printf ( "\n\nwhich=%d ncases=%d trend=%.3lf nreps=%d", which, ncases, save_trend, nreps ) ;

/*
   Initialize
*/

   x = (double *) malloc ( ncases * sizeof(double) ) ;

/*
   Main replication loop
*/

   IS_mean = OOS_mean = 0.0 ;
   for (irep=0 ; irep<nreps ; irep++) {  // Do many trials to get a stable average

      // Generate the in-sample set (log prices)
      trend = save_trend ;
      x[0] = 0.0 ;
      for (i=1 ; i<ncases ; i++) {
         if (i % 50 == 0) // Reverse the trend
            trend = -trend ;
         x[i] = x[i-1] + trend + unifrand() + unifrand() - unifrand() - unifrand() ;
         }

      // Compute optimal parameters, evaluate return with same dataset
      opt_params ( which , ncases , x , &short_lookback , &long_lookback ) ;
      IS_perf = test_system ( ncases , x , short_lookback , long_lookback ) ;

      // Generate the out_of-sample set (log prices)
      trend = save_trend ;
      x[0] = 0.0 ;
      for (i=1 ; i<ncases ; i++) {
         if (i % 50 == 0)
            trend = -trend ;
         x[i] = x[i-1] + trend + unifrand() + unifrand() - unifrand() - unifrand() ;
         }

      // Test the OOS set and cumulate means across replications
      OOS_perf = test_system ( ncases , x , short_lookback , long_lookback ) ;

      IS_mean += IS_perf ;
      OOS_mean += OOS_perf ;
      printf ( "\n%3d: %3d %3d  %8.4lf %8.4lf (%8.4lf)", irep, short_lookback, long_lookback, IS_perf, OOS_perf, IS_perf - OOS_perf ) ;
      } // For irep

   // Done.  Print results and clean up.
   IS_mean /= nreps ;
   OOS_mean /= nreps ;
   printf ( "\nMean IS=%.4lf  OOS=%.4lf  Bias=%.4lf", IS_mean, OOS_mean, IS_mean - OOS_mean ) ;
   _getch () ;  // Wait for user to press a key

   free ( x ) ;

   return 0 ;
}
