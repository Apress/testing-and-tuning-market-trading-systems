/******************************************************************************/
/*                                                                            */
/*  BOUND_MEAN - Use the PER_WHAT system to compare methods for bounding      */
/*               the expected returns computed several ways.                  */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>

#define MKTBUF 2048   /* Alloc for market info in chunks of this many records */
                      /* This is not critical and can be any reasonable vlaue */

double t_CDF ( int ndf , double t ) ;
double inverse_t_CDF ( int ndf , double p ) ;

void boot_conf_pctile ( // Percentile method
   int n ,              // Number of cases in sample
   double *x ,          // Variable in sample
   double (*user_t) (int , double *) , // Compute parameter
   int nboot ,          // Number of bootstrap replications
   double *low2p5 ,     // Output of lower 2.5% bound
   double *high2p5 ,    // Output of upper 2.5% bound
   double *low5 ,       // Output of lower 5% bound
   double *high5 ,      // Output of upper 5% bound
   double *low10 ,      // Output of lower 10% bound
   double *high10 ,     // Output of upper 10% bound
   double *xwork ,      // Work area n long
   double *work2        // Work area nboot long
   ) ;

void boot_conf_BCa (
   int n ,              // Number of cases in sample
   double *x ,          // Variable in sample
   double (*user_t) (int , double * ) , // Compute parameter
   int nboot ,          // Number of bootstrap replications
   double *low2p5 ,     // Output of lower 2.5% bound
   double *high2p5 ,    // Output of upper 2.5% bound
   double *low5 ,       // Output of lower 5% bound
   double *high5 ,      // Output of upper 5% bound
   double *low10 ,      // Output of lower 10% bound
   double *high10 ,     // Output of upper 10% bound
   double *xwork ,      // Work area n long
   double *work2        // Work area nboot long
   ) ;



/*
--------------------------------------------------------------------------------

   Local routine computes optimal lookback and breakout threshold
   for a primitive long-only moving-average breakout system.

--------------------------------------------------------------------------------
*/

double opt_params (
   int nprices ,      // Number of log prices in 'prices'
   double *prices ,   // Log prices
   int max_lookback , // Maximum lookback to use
   int *lookback ,    // Returns optimal MA lookback
   double *thresh ,   // Returns optimal breakout threshold factor
   int *last_pos      // Returns position at end of training set
   )
{
   int i, j, ilook, ibestlook, ithresh, ibestthresh, n_trades ;
   int position, last_position_of_best ;
   double MA_sum, MA_mean, trial_thresh, ret ;
   double best_perf, total_return ;

   best_perf = -1.e60 ;                            // Will be best performance across all trials
   for (ilook=2 ; ilook<=max_lookback ; ilook++) { // Trial MA lookback
      for (ithresh=1 ; ithresh<=10 ; ithresh++) {  // Trial threshold is 0.01 * ithresh

         // We have a pair of parameters (lookback and threshold) to try.
         // Cumulate performance for all valid cases.
         // The index of the first legal bar in prices is max_lookback-1, because we will
         // need max_lookback cases (including the decision bar) in the moving average.
         // We start at the same bar for all lookbacks to make them comparable.
         // We must stop one bar before the end of the price array because we need
         // the next price to compute the return from the decision.

         total_return = 0.0 ;                    // Cumulate total return for this trial
         n_trades = 0 ;                          // Will count trades
         position = 0 ;                          // Current position

         for (i=max_lookback-1 ; i<nprices-1 ; i++) { // Compute performance across history
                                                      // We are making a decision at bar 'i'

            if (i == max_lookback-1) { // Find the moving average for the first valid case.
               MA_sum = 0.0 ;                    // Cumulates MA sum
               for (j=i ; j>i-ilook ; j--)
                  MA_sum += prices[j] ;
               }

            else                                 // Update the moving average
               MA_sum += prices[i] - prices[i-ilook] ;

            MA_mean = MA_sum / ilook ;           // Divide price sum by lookback to get MA
            trial_thresh = 1.0 + 0.01 * ithresh ;

            // We now have the trial threshold and the moving average ending at day i.
            // Make a trade decision.  This is simpler than the algorithm in the book
            // because we do not have to explicitly open and close positions when
            // we are writing our own software and just marking to market.
            // But it is exactly the same algorithm in its effect.
            // If neither the entry test nor the exit test pass, we keep the same position.

            if (prices[i] > trial_thresh * MA_mean)  // Do we satisfy the entry test?
               position = 1 ;

            else if (prices[i] < MA_mean)            // Do we satisfy the exit test?
               position = 0 ;

            if (position)
               ret = prices[i+1] - prices[i] ;       // Return to next bar after decision
            else
               ret = 0.0 ;

            if (position) {
               ++n_trades ;
               total_return += ret ;
               }

            } // For i, summing performance for this trial parameter set

         // We now have the performance figures across the history.
         // Keep track of the best parameters.

         total_return /= n_trades + 1.e-30 ;
         if (total_return > best_perf) {
            best_perf = total_return ;
            ibestlook = ilook ;
            ibestthresh = ithresh ;
            last_position_of_best = position ;
            }
         } // For ithresh, all short-term lookbacks
      } // For ilook, all long-term lookbacks

   *lookback = ibestlook ;
   *thresh = 0.01 * ibestthresh ;
   *last_pos = last_position_of_best ;

   return best_perf ;
}


/*
--------------------------------------------------------------------------------

  Local routine computes return vector for all bars but outputs returns
  according to user's request.

  The first trade decision is made on the last bar of the training set
  and its return is based on the first bar in the test set.
  This first trade decision can depend on the prior position, that
  as of the last trade in the training set.  This happens when neither
  the entry nor the exit rule fires, so we just continue the position.

  The trading system here is long-only, so if you have a short-only system
  or a system that can be either, you will have to slightly modify the
  accounting system.

--------------------------------------------------------------------------------
*/

void comp_return (
   int ret_type ,    // Return type: 0=all bars; 1=bars with open position; 2=completed trades
   int nprices ,     // Number of log prices in 'prices' used only for safety, not algorithm
   double *prices ,  // Log prices
   int istart    ,   // Starting index in OOS test set
   int ntest ,       // Number of OOS test cases
   int lookback ,    // Optimal MA lookback
   double thresh ,   // Optimal breakout threshold factor
   int last_pos ,    // Position in bar prior to test set (typically last training set position)
   int *n_returns ,  // Number of returns in 'returns' array
   double *returns   // Bar returns returned here
   )
{
   int i, j, position, prior_position, nret ;
   double MA_sum, MA_mean, trial_thresh, open_price, ret ;

   nret = 0 ;
   position = last_pos ;           // Current position
   prior_position = 0 ;            // For completed trades, always start out of market
   trial_thresh = 1.0 + thresh ;   // Make it multiplicative for simplicity

   for (i=istart-1 ; i<istart-1+ntest ; i++) {  // Compute returns across test set
                                                // We are making a decision at bar 'i'

      if (i == istart-1) { // Find the moving average for the first valid case.
         MA_sum = 0.0 ;                    // Cumulates MA sum
         for (j=i ; j>i-lookback ; j--) {
            assert ( j >= 0 ) ;
            MA_sum += prices[j] ;
            }
         }

      else                                 // Update the moving average
         MA_sum += prices[i] - prices[i-lookback] ;

      MA_mean = MA_sum / lookback ;        // Divide price sum by lookback to get MA

      // Make a trade decision.  This is simpler than the algorithm in the book
      // because we do not have to explicitly open and close positions when
      // we are writing our own software and just marking to market.
      // But it is exactly the same algorithm in its effect.
      // If the open rule fires, we make sure a position is open (it may already be open).
      // If the exit rule fires, we close the position.
      // If neither rule fires, we maintain the prior position.

      assert ( i+1 < nprices ) ;               // Optional cheap insurance

      if (prices[i] > trial_thresh * MA_mean)  // Do we satisfy the entry test?
         position = 1 ;

      else if (prices[i] < MA_mean)            // Do we satisfy the exit test?
         position = 0 ;

      if (position)
         ret = prices[i+1] - prices[i] ;
      else
         ret = 0.0 ;

      // At this time we know our position and return for this bar.
      // Save (or not) the appropriate outputted return.

      if (ret_type == 0)             // All bars, even those with no position
         returns[nret++] = ret ;

      else if (ret_type == 1) {      // Only bars with a position
         if (position)
            returns[nret++] = ret ;
         }

      else if (ret_type == 2) {      // Completed trades
         if (position  &&  ! prior_position)       // We just opened a trade
            open_price = prices[i] ;
         else if (prior_position  &&  ! position)  // We just closed a trade
            returns[nret++] = prices[i] - open_price ;
         else if (position  &&  i==istart-2+ntest) // Force close at end of data
            returns[nret++] = prices[i+1] - open_price ;
         }


      prior_position = position ;
      } // For i, computing returns across test set

   *n_returns = nret ;
}


/*
--------------------------------------------------------------------------------

   Local routine computes the mean

--------------------------------------------------------------------------------
*/

double find_mean ( int n , double *x )
{
   int i ;
   double sum ;

   sum = 0.0 ;
   for (i=0 ; i<n ; i++)
      sum += x[i] ;

   return sum / n ;
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
   int i, j, nprices, bufcnt, max_lookback, lookback, last_pos, n_returns ;
   int n, train_start, n_train, n_test, n_boot ;
   int nret_open, nret_complete, nret_grouped, crunch ;
   double *prices, *returns_grouped, *returns_open, *returns_complete, thresh, crit, sum, diff ;
   double mean_open, stddev_open, mean_complete, stddev_complete, mean_grouped, stddev_grouped ;
   double *xwork, *work2, high ;
   double t_open, p_open, t_complete, p_complete, t_grouped, p_grouped ;
   double t_lower_open, t_lower_complete, t_lower_grouped ;
   double b1_lower_open, b1_lower_complete, b1_lower_grouped ;
   double b2_lower_open, b2_lower_complete, b2_lower_grouped ;
   double b3_lower_open, b3_lower_complete, b3_lower_grouped ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 6) {
      printf ( "\nUsage: BOUND_MEAN  max_lookback  n_train  n_test  n_boot  filename" ) ;
      printf ( "\n  max_lookback - Maximum moving-average lookback" ) ;
      printf ( "\n  n_train - Number of bars in training set (much greater than max_lookback)" ) ;
      printf ( "\n  n_test - Number of bars in test set" ) ;
      printf ( "\n  n_boot - Number of bootstrap reps" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   max_lookback = atoi ( argv[1] ) ;
   n_train = atoi ( argv[2] ) ;
   n_test = atoi ( argv[3] ) ;
   n_boot = atoi ( argv[4] ) ;
   strcpy_s ( filename , argv[5] ) ;
#else
   max_lookback = 100 ;
   n_train = 2000 ;
   n_test = 1000 ;
   n_boot = 100000 ;
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

   returns_open = (double *) malloc ( 3 * nprices * sizeof(double) ) ;
   if (returns_open == NULL) {
      free ( prices ) ;
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      _getch () ;  // Wait for user to press a key
      exit ( 1 ) ;
      }

   returns_complete = returns_open + nprices ;
   returns_grouped = returns_complete + nprices ;

   xwork = (double *) malloc ( (nprices + n_boot) * sizeof(double) ) ;
   if (xwork == NULL) {
      free ( prices ) ;
      free ( returns_open ) ;
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      _getch () ;  // Wait for user to press a key
      exit ( 1 ) ;
      }

   work2 = xwork + nprices ;

   train_start = 0 ;      // Starting index of training set
   nret_open = nret_complete = nret_grouped = 0 ;

/*
   Do walkforward
*/

   for (;;) {

      // Train

      crit = opt_params ( n_train , prices + train_start ,
                          max_lookback , &lookback , &thresh , &last_pos ) ;
      printf ( "\n IS at %d  Lookback=%d  Thresh=%.3lf  Crit=%.3lf",
               train_start, lookback, thresh, crit ) ;

      n = n_test ;    // Test this many cases
      if (n > nprices - train_start - n_train) // Don't go past the end of history
         n = nprices - train_start - n_train ;

      // Test with each of the three return types

      comp_return ( 0 , nprices , prices , train_start + n_train , n , lookback ,
                    thresh , last_pos , &n_returns , returns_grouped + nret_grouped ) ;
      nret_grouped += n_returns ;

      printf ( "\nOOS 0 testing %d from %d had %d returns, total=%d",
               n, train_start + n_train, n_returns, nret_grouped ) ;

      comp_return ( 1 , nprices , prices , train_start + n_train , n , lookback ,
                    thresh , last_pos , &n_returns , returns_open + nret_open ) ;
      nret_open += n_returns ;

      printf ( "\nOOS 1 testing %d from %d had %d returns, total=%d",
               n, train_start + n_train, n_returns, nret_open ) ;

      comp_return ( 2 , nprices , prices , train_start + n_train , n , lookback ,
                    thresh , last_pos , &n_returns , returns_complete + nret_complete ) ;
      nret_complete += n_returns ;

      printf ( "\nOOS 2 testing %d from %d had %d returns, total=%d",
               n, train_start + n_train, n_returns, nret_complete ) ;

      // Advance fold window; quit if done
      train_start += n ;
      if (train_start + n_train >= nprices)
         break ;
      }

/*
   Crunch the grouped returns
*/

   crunch = 10 ;  // Change this to whatever you wish
   n_returns = (nret_grouped + crunch - 1) / crunch ;  // This many returns after crunching

   for (i=0 ; i<n_returns ; i++) {            // Each crunched return
      n = crunch ;                            // Normally this many in group
      if (i*crunch+n > nret_grouped)          // May run short in last group
         n = nret_grouped - i*crunch ;        // This many in last group
      sum = 0.0 ;
      for (j=i*crunch ; j<i*crunch+n ; j++)   // Sum all in this gorup
         sum += returns_grouped[j] ;
      returns_grouped[i] = sum / n ;          // Compute mean per group
      }

   nret_grouped = n_returns ;

/*
   Compute and print OOS performance
*/

   printf ( "\n\nnprices=%d  max_lookback=%d  n_train=%d  n_test=%d",
            nprices, max_lookback, n_train, n_test ) ;

   mean_open = 0.0 ;
   for (i=0 ; i<nret_open ; i++)
      mean_open += returns_open[i] ;
   mean_open /= (nret_open + 1.e-60) ;
   stddev_open = 0.0 ;
   for (i=0 ; i<nret_open ; i++) {
      diff = returns_open[i] - mean_open ;
      stddev_open += diff * diff ;
      }
   if (nret_open > 1) {
      stddev_open = sqrt ( stddev_open / (nret_open - 1) ) ;
      t_open = sqrt((double) nret_open) * mean_open / (stddev_open + 1.e-20) ;
      p_open = 1.0 - t_CDF ( nret_open-1 , t_open ) ;
      t_lower_open = mean_open - stddev_open / sqrt((double) nret_open) * inverse_t_CDF ( nret_open-1 , 0.9 ) ;
      }
   else {
      stddev_open = t_open = 0.0 ;
      p_open = 1.0 ;
      t_lower_open = 0.0 ;
      }
   printf ( "\nOOS mean return per open-trade bar (times 25200) = %.5lf\n  StdDev = %.5lf  t = %.2lf  p = %.4lf  lower = %.5lf  nret=%d",
            25200 * mean_open, 25200 * stddev_open, t_open, p_open, 25200 * t_lower_open, nret_open ) ;

   mean_complete = 0.0 ;
   for (i=0 ; i<nret_complete ; i++)
      mean_complete += returns_complete[i] ;
   mean_complete /= (nret_complete + 1.e-60) ;
   stddev_complete = 0.0 ;
   for (i=0 ; i<nret_complete ; i++) {
      diff = returns_complete[i] - mean_complete ;
      stddev_complete += diff * diff ;
      }
   if (nret_complete > 1) {
      stddev_complete = sqrt ( stddev_complete / (nret_complete - 1) ) ;
      t_complete = sqrt((double) nret_complete) * mean_complete / (stddev_complete + 1.e-20) ;
      p_complete = 1.0 - t_CDF ( nret_complete-1 , t_complete ) ;
      t_lower_complete = mean_complete - stddev_complete / sqrt((double) nret_complete) * inverse_t_CDF ( nret_complete-1 , 0.9 ) ;
      }
   else {
      stddev_complete = t_complete = 0.0 ;
      p_complete = 1.0 ;
      t_lower_complete = 0.0 ;
      }
   printf ( "\nOOS mean return per complete trade (times 1000) = %.5lf\n  StdDev = %.5lf  t = %.2lf  p = %.4lf  lower = %.5lf  nret=%d",
            1000 * mean_complete, 1000 * stddev_complete, t_complete, p_complete, 1000 * t_lower_complete, nret_complete ) ;

   mean_grouped = 0.0 ;
   for (i=0 ; i<nret_grouped ; i++)
      mean_grouped += returns_grouped[i] ;
   mean_grouped /= (nret_grouped + 1.e-60) ;
   stddev_grouped = 0.0 ;
   for (i=0 ; i<nret_grouped ; i++) {
      diff = returns_grouped[i] - mean_grouped ;
      stddev_grouped += diff * diff ;
      }
   if (nret_grouped > 1) {
      stddev_grouped = sqrt ( stddev_grouped / (nret_grouped - 1) ) ;
      t_grouped = sqrt((double) nret_grouped) * mean_grouped / (stddev_grouped + 1.e-20) ;
      p_grouped = 1.0 - t_CDF ( nret_grouped-1 , t_grouped ) ;
      t_lower_grouped = mean_grouped - stddev_grouped / sqrt((double) nret_grouped) * inverse_t_CDF ( nret_grouped-1 , 0.9 ) ;
      }
   else {
      stddev_grouped = t_grouped = 0.0 ;
      p_grouped = 1.0 ;
      t_lower_grouped = 0.0 ;
      }
   printf ( "\nOOS mean return per %d-bar group (times 25200) = %.5lf\n  StdDev = %.5lf  t = %.2lf  p = %.4lf  lower = %.5lf  nret=%d",
            crunch, 25200 * mean_grouped, 25200 * stddev_grouped, t_grouped, p_grouped, 25200 * t_lower_grouped, nret_grouped ) ;

   if (nret_open < 2  ||  nret_complete < 2  ||  nret_grouped < 2) {
      printf ( "\n\nBootstraps skipped due to too few returns" ) ;
      goto FINISH ;
      }


/*
   Do bootstraps
*/

   printf ( "\n\nDoing bootstrap 1 of 6..." ) ;
   boot_conf_pctile ( nret_open , returns_open , find_mean , n_boot , 
                      &sum , &sum , &sum , &sum , &b1_lower_open , &high ,
                      xwork , work2 ) ;
   b2_lower_open = 2.0 * mean_open - high ;

   printf ( "\nDoing bootstrap 2 of 6..." ) ;
   boot_conf_BCa ( nret_open , returns_open , find_mean , n_boot , 
                   &sum , &sum , &sum , &sum , &b3_lower_open , &high ,
                   xwork , work2 ) ;

   printf ( "\nDoing bootstrap 3 of 6..." ) ;
   boot_conf_pctile ( nret_complete , returns_complete , find_mean , n_boot , 
                      &sum , &sum , &sum , &sum , &b1_lower_complete , &high ,
                      xwork , work2 ) ;
   b2_lower_complete = 2.0 * mean_complete - high ;

   printf ( "\nDoing bootstrap 4 of 6..." ) ;
   boot_conf_BCa ( nret_complete , returns_complete , find_mean , n_boot , 
                   &sum , &sum , &sum , &sum , &b3_lower_complete , &high ,
                   xwork , work2 ) ;

   printf ( "\nDoing bootstrap 5 of 6..." ) ;
   boot_conf_pctile ( nret_grouped , returns_grouped , find_mean , n_boot , 
                      &sum , &sum , &sum , &sum , &b1_lower_grouped , &high ,
                      xwork , work2 ) ;
   b2_lower_grouped = 2.0 * mean_grouped - high ;

   printf ( "\nDoing bootstrap 6 of 6..." ) ;
   boot_conf_BCa ( nret_grouped , returns_grouped , find_mean , n_boot , 
                   &sum , &sum , &sum , &sum , &b3_lower_grouped , &high ,
                   xwork , work2 ) ;

   printf ( "\n\n90 percent lower confidence bounds" ) ;
   printf ( "\n            Open posn   Complete   Grouped" ) ;
   printf ( "\nStudent's t  %7.4lf    %7.4lf    %7.4lf",
            25200 * t_lower_open, 1000 * t_lower_complete, 25200 * t_lower_grouped ) ;
   printf ( "\nPercentile   %7.4lf    %7.4lf    %7.4lf",
            25200 * b1_lower_open, 1000 * b1_lower_complete, 25200 * b1_lower_grouped ) ;
   printf ( "\nPivot        %7.4lf    %7.4lf    %7.4lf",
            25200 * b2_lower_open, 1000 * b2_lower_complete, 25200 * b2_lower_grouped ) ;
   printf ( "\nBCa          %7.4lf    %7.4lf    %7.4lf",
            25200 * b3_lower_open, 1000 * b3_lower_complete, 25200 * b3_lower_grouped ) ;

FINISH:
   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   free ( prices ) ;
   free ( returns_open ) ;
   free ( xwork ) ;

   exit ( 0 ) ;
}
