/******************************************************************************/
/*                                                                            */
/*  SelBias - Explore selection bias                                          */
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
   int which ,        // 0=mean return; 1=profit factor; 2=Sharpe ratio
   int long_v_short , // Long only if nonzero, else short-only
   int ncases ,       // Number of log prices in X
   double *x ,        // Log prices
   int *short_term ,  // Returns optimal short-term lookback
   int *long_term     // Returns optimal long-term lookback
   )
{
   int i, j, ishort, ilong, ibestshort, ibestlong, traded, n_trades ;
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
         n_trades = 0 ;                          // Will count trades

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

            traded = 0 ;                                            // Did we do a trade?
            if (long_v_short  &&  short_mean > long_mean) {         // Long position
               ret = x[i+1] - x[i] ;
               traded = 1 ;
               }
            else if (! long_v_short  &&  short_mean < long_mean) {  // Short position
               ret = x[i] - x[i+1] ;
               traded = 1 ;
               }

            if (traded) {
               ++n_trades ;
               total_return += ret ;
               sum_squares += ret * ret ;
               if (ret > 0.0)
                  win_sum += ret ;
               else
                  lose_sum -= ret ;
               }

            } // For i, summing performance for this trial

         // We now have the performance figures across the history
         // Keep track of the best lookbacks

         if (which == 0) {      // Mean return criterion
            total_return /= n_trades + 1.e-30 ;
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
            total_return /= n_trades + 1.e-30 ;   // Now mean return
            sum_squares /= n_trades + 1.e-30 ;
            sum_squares -= total_return * total_return ;  // Variance (may be zero!)
            if (sum_squares < 1.e-20)  // Must not divide by zero or take sqrt of negative
               sum_squares = 1.e-20 ;
            sr = total_return / sqrt ( sum_squares ) ;
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
   int long_v_short , // Long only if nonzero, else short-only
   int ncases ,
   double *x ,
   int short_term ,
   int long_term
   )
{
   int i, j, n_trades ;
   double sum, short_mean, long_mean ;

   sum = 0.0 ;                          // Cumulate performance for this trial
   n_trades = 0 ;
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

      if (long_v_short  &&  short_mean > long_mean) {         // Long position
         sum += x[i+1] - x[i] ;
         ++n_trades ;
         }
      else if (! long_v_short  &&  short_mean < long_mean) {  // Short position
         sum -= x[i+1] - x[i] ;
         ++n_trades ;
         }
      } // For i, summing performance for this trial

   sum /= n_trades + 1.e-30 ;     // Mean return across the history we just tested

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
   int i, which, ncases, irep, nreps, L_short_lookback, L_long_lookback, S_short_lookback, S_long_lookback ;
   double save_trend, trend, *x, L_IS_perf, S_IS_perf, L_OOS_perf, S_OOS_perf, OOS_mean ;
   double Bias, Bias_mean, OOS_perf, L_IS_mean, S_IS_mean, L_OOS_mean, S_OOS_mean, Bias_SS, t ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 5) {
      printf ( "\nUsage: SelBias  which  ncases trend  nreps" ) ;
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
      printf ( "\nUsage: SelBias  which  ncases trend  nreps" ) ;
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

   L_IS_mean = S_IS_mean = L_OOS_mean = S_OOS_mean = OOS_mean = Bias_mean = Bias_SS = 0.0 ;

   for (irep=0 ; irep<nreps ; irep++) {  // Do many trials to get a stable average

      // Generate the in-sample set (log prices)
      trend = save_trend ;
      x[0] = 0.0 ;
      for (i=1 ; i<ncases ; i++) {
         if ((i+1) % 50 == 0)   // Reverse the trend every 50 days
            trend = -trend ;
         x[i] = x[i-1] + trend + unifrand() + unifrand() - unifrand() - unifrand() ;
         }

      // Compute optimal parameters, evaluate return with same dataset
      // The first pair of lines below is for the long-only model, second pair short-only

      opt_params ( which , 1 , ncases , x , &L_short_lookback , &L_long_lookback ) ;
      L_IS_perf = test_system ( 1 , ncases , x , L_short_lookback , L_long_lookback ) ;

      opt_params ( which , 0 , ncases , x , &S_short_lookback , &S_long_lookback ) ;
      S_IS_perf = test_system ( 0 , ncases , x , S_short_lookback , S_long_lookback ) ;

      // Generate the first out_of-sample set (log prices)
      // This will give us the performance results on which our choice of model is based

      trend = save_trend ;
      x[0] = 0.0 ;
      for (i=1 ; i<ncases ; i++) {
         if ((i+1) % 50 == 0)
            trend = -trend ;
         x[i] = x[i-1] + trend + unifrand() + unifrand() - unifrand() - unifrand() ;
         }

      // Test this first OOS set and cumulate means across replications
      // We will compare L_OOS_perf with S_OOS_perf to choose the best model for the final test

      L_OOS_perf = test_system ( 1 , ncases , x , L_short_lookback , L_long_lookback ) ;
      S_OOS_perf = test_system ( 0 , ncases , x , S_short_lookback , S_long_lookback ) ;

      L_IS_mean += L_IS_perf ;
      L_OOS_mean += L_OOS_perf ;
      S_IS_mean += S_IS_perf ;
      S_OOS_mean += S_OOS_perf ;
      printf ( "\n%3d: %3d %3d %3d %3d  %8.4lf %8.4lf (%8.4lf)  %8.4lf %8.4lf (%8.4lf)",
               irep, L_short_lookback, L_long_lookback, S_short_lookback, S_long_lookback,
               L_IS_perf, L_OOS_perf, L_IS_perf - L_OOS_perf, S_IS_perf, S_OOS_perf, S_IS_perf - S_OOS_perf ) ;

      // Generate the second out_of-sample set (log prices)
      // This is the 'ultimate' OOS set, which has selection bias removed

      trend = save_trend ;
      x[0] = 0.0 ;
      for (i=1 ; i<ncases ; i++) {
         if ((i+1) % 50 == 0)
            trend = -trend ;
         x[i] = x[i-1] + trend + unifrand() + unifrand() - unifrand() - unifrand() ;
         }

      // Test this second OOS set and cumulate means across replications
      // We choose either the long or the short model, depending on which
      // did better on the first OOS set

      if (L_OOS_perf > S_OOS_perf) {
         OOS_perf = test_system ( 1 , ncases , x , L_short_lookback , L_long_lookback ) ;
         Bias = L_OOS_perf - OOS_perf ;  // This is the selection bias for this replication
         }
      else {
         OOS_perf = test_system ( 0 , ncases , x , S_short_lookback , S_long_lookback ) ;
         Bias = S_OOS_perf - OOS_perf ;  // This is the selection bias for this replication
         }

      Bias_mean += Bias ;
      Bias_SS += Bias * Bias ;  // We'll need this for t-test
      OOS_mean += OOS_perf ;    // This is the final OOS performance, with both training and selection bias removed
      printf ( "\n     OOS_perf=%8.4lf  Bias=%8.4lf", OOS_perf, Bias ) ;
      } // For irep

   // Done.  Print results and clean up.

   L_IS_mean /= nreps ;   // These are for computing training bias
   L_OOS_mean /= nreps ;  // We compute long and short separately
   S_IS_mean /= nreps ;   // Because in general different competing models
   S_OOS_mean /= nreps ;  // can have different training bias

   printf ( "\n\nLong training bias = %.4lf  short = %.4lf",
            L_IS_mean - L_OOS_mean, S_IS_mean - S_OOS_mean ) ;

   OOS_mean /= nreps ;
   Bias_mean /= nreps ;
   Bias_SS /= nreps ;
   Bias_SS -= Bias_mean * Bias_mean ;  // This is the variance of the bias across replications
   if (Bias_SS < 1.e-20)  // Don't divide by zero or take sqrt of a negative number
      Bias_SS = 1.e-20 ;
   t = sqrt((double) nreps) * Bias_mean / sqrt ( Bias_SS ) ;  // Compute t-score

   printf ( "\nOOS=%.4lf  Selection bias=%.4lf  t=%.3lf", OOS_mean, Bias_mean, t ) ;
   _getch () ;  // Wait for user to press a key

   free ( x ) ;

   return 0 ;
}
