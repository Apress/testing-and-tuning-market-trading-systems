/*****************************************************************************/
/*                                                                           */
/*  BND_RET - Use a primitive moving-average-crossover system to demonstrate */
/*            bounding future returns                                        */
/*                                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>

void qsortd ( int istart , int istop , double *x ) ;
double orderstat_tail ( int n , double q , int m ) ;
double quantile_conf ( int n , int m , double conf ) ;

#define MKTBUF 2048   /* Alloc for market info in chunks of this many records */
                      /* This is not critical and can be any reasonable vlaue */


/*
--------------------------------------------------------------------------------

   Local routine computes optimal short-term and long-term lookbacks
   for a primitive moving-average crossover system

--------------------------------------------------------------------------------
*/

double opt_params (
   int ncases ,       // Number of log prices in X
   int max_lookback , // Maximum lookback to try
   double *x ,        // Log prices
   int *short_term ,  // Returns optimal short-term lookback
   int *long_term     // Returns optimal long-term lookback
   )
{
   int i, j, ishort, ilong, ibestshort, ibestlong ;
   double short_sum, long_sum, short_mean, long_mean, total_return, best_perf, ret ;

   best_perf = -1.e60 ;                           // Will be best performance across all trials
   for (ilong=2 ; ilong<max_lookback ; ilong++) { // Trial long-term lookback
      for (ishort=1 ; ishort<ilong ; ishort++) {  // Trial short-term lookback

         // We have a pair of lookbacks to try.
         // Cumulate performance for all valid cases

         total_return = 0.0 ;                    // Cumulate total return for this trial

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
            } // For i, summing performance for this trial

         // We now have the performance figures across the history
         // Keep track of the best lookbacks

         total_return /= ncases - ilong ;
         if (total_return > best_perf) {
            best_perf = total_return ;
            ibestshort = ishort ;
            ibestlong = ilong ;
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
   int i, j, n ;
   double sum, short_mean, long_mean ;

   n = ncases ;
   sum = 0.0 ;                         // Cumulate performance for this trial
   for (i=long_term-1 ; n>0 ; i++) {   // Sum performance across history
      short_mean = 0.0 ;               // Cumulates short-term lookback sum
      for (j=i ; j>i-short_term ; j--)
         short_mean += x[j] ;
      long_mean = short_mean ;         // Cumulates long-term lookback sum
      while (j>i-long_term)
         long_mean += x[j--] ;
      short_mean /= short_term ;
      long_mean /= long_term ;
      // We now have the short-term and long-term means ending at day i
      // Take our position and cumulate return
      if (short_mean > long_mean)      // Long position
         sum += x[i+1] - x[i] ;
      else if (short_mean < long_mean) // Short position
         sum -= x[i+1] - x[i] ;
      --n ;
      } // For i, summing performance for this trial
   sum /= ncases ;              // Mean return across the history we just tested

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
   int i, nprices, bufcnt, max_lookback, long_lookback, short_lookback, n_returns ;
   int n, train_start, n_train, n_test, lower_bound_m, upper_bound_m ;
   double IS, OOS, *prices, *returns, total ;
   double lower_bound, upper_bound, lower_fail_rate, upper_fail_rate ;
   double lower_bound_opt_q, lower_bound_pes_q, lower_bound_opt_prob, lower_bound_pes_prob ;
   double upper_bound_opt_q, upper_bound_pes_q, upper_bound_opt_prob, upper_bound_pes_prob ;
   double p_of_q, lower_bound_p_of_q_opt_q, lower_bound_p_of_q_pes_q, upper_bound_p_of_q_opt_q, upper_bound_p_of_q_pes_q ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 8) {
      printf ( "\nUsage: BND_RET  max_lookback  n_train  n_test  lower_fail  upper_fail  p_of_q  filename" ) ;
      printf ( "\n  max_lookback - Maximum moving-average lookback" ) ;
      printf ( "\n  n_train - Number of bars in training set (much greater than max_lookback)" ) ;
      printf ( "\n  n_test - Number of bars in test set" ) ;
      printf ( "\n  lower_fail - Lower bound failure rate (often 0.01-0.1)" ) ;
      printf ( "\n  upper_fail - Upper bound failure rate (often 0.1-0.5)" ) ;
      printf ( "\n  p_of_q - Probability of bad bound (often 0.01-0.1)" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   max_lookback = atoi ( argv[1] ) ;
   n_train = atoi ( argv[2] ) ;
   n_test = atoi ( argv[3] ) ;
   lower_fail_rate = atof ( argv[4] ) ;
   upper_fail_rate = atof ( argv[5] ) ;
   p_of_q = atof ( argv[6] ) ;
   strcpy_s ( filename , argv[7] ) ;
#else
   max_lookback = 100 ;
   n_train = 1000 ;
   n_test = 63 ;
   lower_fail_rate = 0.1 ;
   upper_fail_rate = 0.4 ;
   p_of_q = 0.05 ;
   strcpy_s ( filename , "E:\\MarketDataAssorted\\INDEXES\\$OEX.TXT" ) ;
#endif

   if (n_train - max_lookback < 10) {
      printf ( "\nERROR... n_train must be at least 10 greater than max_lookback" ) ;
      exit ( 1 ) ;
      }


/*
   Read market prices
*/

   if (fopen_s ( &fp, filename , "rt" )) {
      printf ( "\n\nCannot open market history file %s", filename ) ;
      exit ( 1 ) ;
      }

   prices = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   if (prices == NULL) {
      printf ( "\n\nInsufficient memory reading market history file %s  Press any key...", filename ) ;
      _getch () ;  // Wait for user to press a key
      fclose ( fp ) ;
      exit ( 1 ) ;
      }

   bufcnt = MKTBUF ;  // There are this many record slots available now

   printf ( "\nReading market file..." ) ;

   nprices = 0 ;    // Counts lines (prices) read

   for (;;) {

      if (feof ( fp )                          // If end of file
       || (fgets ( line , 256 , fp ) == NULL)  // Or unable to read line
       || (strlen ( line ) < 2))               // Or empty line
         break ;                               // We are done reading price history

      if (ferror ( fp )) {                     // If an error reading file
         fclose ( fp ) ;                       // Quit immediately
         free ( prices ) ;
         printf ( "\nError reading line %d of file %s", nprices+1, filename ) ;
         exit ( 1 ) ;
         }

      if (! bufcnt) {  // Allocate a new memory block if needed
         prices = (double *) realloc ( prices , (nprices+MKTBUF) * sizeof(double) ) ;
         if (prices == NULL) {
            fclose ( fp ) ;
            printf ( "\n\nInsufficient memory reading market history file %s  Press any key...", filename ) ;
            _getch () ;  // Wait for user to press a key
            exit ( 1 ) ;
            } // If insufficient memory
         bufcnt = MKTBUF ;  // There are this many new record slots available now
         } // If allocating new block

      // Parse the date and do a crude sanity check

      for (i=0 ; i<8 ; i++) {
         if ((line[i] < '0')  ||  (line[i] > '9')) {
            fclose ( fp ) ;
            free ( prices ) ;
            printf ( "\nInvalid date reading line %d of file %s", nprices+1, filename ) ;
            exit ( 1 ) ;
            }
         }

      // Parse the price

      cptr = line + 9 ;  // Price is in this column or beyond
                         // (Next loop allows price to start past this)

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      prices[nprices] = atof ( cptr ) ;
      if (prices[nprices] > 0.0)                     // Always true, but avoid disaster
         prices[nprices] = log ( prices[nprices] ) ;
      ++nprices  ;
      --bufcnt ;           // One less slot remains

      } // For all lines

   fclose ( fp ) ;

   printf ( "\nMarket price history read" ) ;


/*
   The market data is read.  Initialize for walkforward.
*/

   if (n_train + n_test > nprices) {
      free ( prices ) ;
      printf ( "\nERROR... n_train + n_test must not exceed n_prices.  Press any key..." ) ;
      _getch () ;  // Wait for user to press a key
      exit ( 1 ) ;
      }

   returns = (double *) malloc ( nprices * sizeof(double) ) ;
   if (returns == NULL) {
      free ( prices ) ;
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      _getch () ;  // Wait for user to press a key
      exit ( 1 ) ;
      }

   train_start = 0 ;      // Starting index of training set
   n_returns = 0 ;        // Will count returns (after grouping)
   total = 0.0 ;          // Sums returns for user's edification


/*
   Do walkforward
*/

   for (;;) {

      IS = opt_params ( n_train , max_lookback , prices + train_start , &short_lookback , &long_lookback ) ;
      IS *= 25200 ;  // Approximately annualize
      printf ( "\n\nIS = %.3lf at %d  Lookback=%d %d", IS, train_start, short_lookback, long_lookback ) ;

      n = n_test ;    // Test this many cases
      if (n > nprices - train_start - n_train) // Don't go past the end of history
         n = nprices - train_start - n_train ;

      OOS = test_system ( n , prices + train_start + n_train - long_lookback , short_lookback , long_lookback ) ;
      OOS *= 25200 ;  // Approximately annualize
      printf ( "\nOOS = %.3lf at %d", OOS, train_start + n_train ) ;

      returns[n_returns++] = OOS ;
      total += OOS ;

      // Advance fold window; quit if done
      train_start += n ;
      if (train_start + n_train >= nprices)
         break ;
      }

   printf ( "\n\nAll returns are approximately annualized by multiplying by 25200" ) ;
   printf ( "\nmean OOS = %.3lf with %d returns", total / n_returns, n_returns ) ;

/*
   Do return bounding
*/

   qsortd ( 0 , n_returns-1 , returns ) ;

   lower_bound_m = (int) (lower_fail_rate * (n_returns + 1) ) ;
   if (lower_bound_m < 1)
      lower_bound_m = 1 ;

   lower_bound = returns[lower_bound_m-1] ;

   upper_bound_m = (int) (upper_fail_rate * (n_returns + 1) ) ;
   if (upper_bound_m < 1)
      upper_bound_m = 1 ;

   upper_bound = returns[n_returns-upper_bound_m] ;

   lower_bound_opt_q = 0.9 * lower_fail_rate ;  // This is an arbitrary choice which could be user input
   lower_bound_pes_q = 1.1 * lower_fail_rate ;

   upper_bound_opt_q = 0.9 * upper_fail_rate ;
   upper_bound_pes_q = 1.1 * upper_fail_rate ;

   lower_bound_opt_prob = 1.0 - orderstat_tail ( n_returns , lower_bound_opt_q , lower_bound_m ) ;
   lower_bound_pes_prob = orderstat_tail ( n_returns , lower_bound_pes_q , lower_bound_m ) ;

   upper_bound_opt_prob = 1.0 - orderstat_tail ( n_returns , upper_bound_opt_q , upper_bound_m ) ;
   upper_bound_pes_prob = orderstat_tail ( n_returns , upper_bound_pes_q , upper_bound_m ) ;

   lower_bound_p_of_q_opt_q = quantile_conf ( n_returns , lower_bound_m , 1.0 - p_of_q ) ;
   lower_bound_p_of_q_pes_q = quantile_conf ( n_returns , lower_bound_m , p_of_q ) ;

   upper_bound_p_of_q_opt_q = quantile_conf ( n_returns , upper_bound_m , 1.0 - p_of_q ) ;
   upper_bound_p_of_q_pes_q = quantile_conf ( n_returns , upper_bound_m , p_of_q ) ;

   printf ( "\n\nThe LOWER bound on future returns is %.3lf", lower_bound ) ;
   printf ( "\nIt has an expected user-specified failure rate of %.2lf %%", 100 * lower_fail_rate ) ;
   printf ( "\n  (This is the percent of future returns less than the lower bound.)" ) ;

   printf ( "\n\nWe may take an optimistic view: the lower bound is too low." ) ;
   printf ( "\n  (This results in a lower failure rate.)" ) ;
   printf ( "\nThe probability is %.4lf that the true failure rate is %.2lf %% or less",
            lower_bound_opt_prob, 100 * lower_bound_opt_q ) ;
   printf ( "\nThe probability is %.4lf that the true failure rate is %.2lf %% or less",
            p_of_q, 100 * lower_bound_p_of_q_opt_q ) ;

   printf ( "\n\nWe may take a pessimistic view: the lower bound is too high." ) ;
   printf ( "\n  (This results in a higher failure rate.)" ) ;
   printf ( "\nThe probability is %.4lf that the true failure rate is %.2lf %% or more",
            lower_bound_pes_prob, 100 * lower_bound_pes_q ) ;
   printf ( "\nThe probability is %.4lf that the true failure rate is %.2lf %% or more",
            p_of_q, 100 * lower_bound_p_of_q_pes_q ) ;


   printf ( "\n\nThe UPPER bound on future returns is %.3lf", upper_bound ) ;
   printf ( "\nIt has an expected user-specified failure rate of %.2lf %%", 100 * upper_fail_rate ) ;
   printf ( "\n  (This is the percent of future returns greater than the upper bound.)" ) ;

   printf ( "\n\nWe may take an optimistic view: the upper bound is too high." ) ;
   printf ( "\n  (This results in a lower failure rate.)" ) ;
   printf ( "\nThe probability is %.4lf that the true failure rate is %.2lf %% or less",
            upper_bound_opt_prob, 100 * upper_bound_opt_q ) ;
   printf ( "\nThe probability is %.4lf that the true failure rate is %.2lf %% or less",
            p_of_q, 100 * upper_bound_p_of_q_opt_q ) ;

   printf ( "\n\nWe may take a pessimistic view: the upper bound is too low." ) ;
   printf ( "\n  (This results in a higher failure rate.)" ) ;
   printf ( "\nThe probability is %.4lf that the true failure rate is %.2lf %% or more",
            upper_bound_pes_prob, 100 * upper_bound_pes_q ) ;
   printf ( "\nThe probability is %.4lf that the true failure rate is %.2lf %% or more",
            p_of_q, 100 * upper_bound_p_of_q_pes_q ) ;

   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   free ( prices ) ;
   free ( returns ) ;
   exit ( 0 ) ;
}
