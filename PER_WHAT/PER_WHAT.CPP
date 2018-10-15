/******************************************************************************/
/*                                                                            */
/*  PER_WHAT - Demonstrate different ways to compute OOS returns              */
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



/*
--------------------------------------------------------------------------------

   Local routine computes optimal lookback and breakout threshold
   for a primitive long-only moving-average breakout system.
   The caller can choose which of three optimization criteria to use.

--------------------------------------------------------------------------------
*/

double opt_params (
   int which_crit ,   // 0=mean return per bar; 1=profit factor; 2=Sharpe ratio
   int all_bars ,     // Include return of all bars, even those with no position
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
   double MA_sum, MA_mean, trial_thresh, ret, win_sum, lose_sum, sum_squares, sr ;
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
         win_sum = lose_sum = 1.e-60 ;           // Cumulates for profit factor
         sum_squares = 1.e-60 ;                  // Cumulates for Sharpe ratio
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

            if (all_bars  ||  position) {
               ++n_trades ;
               total_return += ret ;
               sum_squares += ret * ret ;
               if (ret > 0.0)
                  win_sum += ret ;
               else
                  lose_sum -= ret ;
               }

            } // For i, summing performance for this trial parameter set

         // We now have the performance figures across the history.
         // Keep track of the best parameters.

         if (which_crit == 0) {      // Mean return criterion
            total_return /= n_trades + 1.e-30 ;
            if (total_return > best_perf) {
               best_perf = total_return ;
               ibestlook = ilook ;
               ibestthresh = ithresh ;
               last_position_of_best = position ;
               }
            }

         else if (which_crit == 1  &&  win_sum / lose_sum > best_perf) { // Profit factor criterion
            best_perf = win_sum / lose_sum ;
            ibestlook = ilook ;
            ibestthresh = ithresh ;
            last_position_of_best = position ;
            }

         else if (which_crit == 2) { // Sharpe ratio criterion
            total_return /= n_trades + 1.e-30 ;   // Now mean return
            sum_squares /= n_trades + 1.e-30 ;
            sum_squares -= total_return * total_return ;  // Variance (may be zero!)
            if (sum_squares < 1.e-20)  // Must not divide by zero or take sqrt of negative
               sum_squares = 1.e-20 ;
            sr = total_return / sqrt ( sum_squares ) ;
            if (sr > best_perf) { // Sharpe ratio
               best_perf = sr ;
               ibestlook = ilook ;
               ibestthresh = ithresh ;
               last_position_of_best = position ;
               }
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

   Main routine

--------------------------------------------------------------------------------
*/

int main (
   int argc ,    // Number of command line arguments (includes prog name)
   char *argv[]  // Arguments (prog name is argv[0])
   )
{
   int i, nprices, all_bars, bufcnt, which_crit, max_lookback, lookback, last_pos, n_returns ;
   int n, mult, train_start, n_train, n_test, nret, ret_type ;
   double *prices, *returns, thresh, crit, sum, sum_squares, win_sum, lose_sum ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 8) {
      printf ( "\nUsage: PER_WHAT  which_crit  all_bars  ret_type  max_lookback  n_train  n_test  filename" ) ;
      printf ( "\n  which_crit - 0=mean return; 1=profit factor; 2=Sharpe ratio" ) ;
      printf ( "\n  all_bars - Training: Include all bars in return, even those with no position?" ) ;
      printf ( "\n  ret_type - Testing: 0=all bars; 1=bars with position open; 2=completed trades" ) ;
      printf ( "\n  max_lookback - Maximum moving-average lookback" ) ;
      printf ( "\n  n_train - Number of bars in training set (much greater than max_lookback)" ) ;
      printf ( "\n  n_test - Number of bars in test set" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   which_crit = atoi ( argv[1] ) ;
   all_bars = atoi ( argv[2] ) ;
   ret_type = atoi ( argv[3] ) ;
   max_lookback = atoi ( argv[4] ) ;
   n_train = atoi ( argv[5] ) ;
   n_test = atoi ( argv[6] ) ;
   strcpy_s ( filename , argv[7] ) ;
#else
   which_crit = 1 ;
   all_bars = 0 ;
   ret_type = 2 ;
   max_lookback = 100 ;
   n_train = 2000 ;
   n_test = 1000 ;
   strcpy_s ( filename , "E:\\MarketDataAssorted\\INDEXES\\$SPX.TXT" ) ;
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


   if (which_crit == 0) {
      mult = 25200 ;      // Scale mean return to be roughly annualized percent
      printf ( "\nMean return criterion will be multiplied by 25200 in all results" ) ;
      }
   else
      mult = 1 ;

   train_start = 0 ;      // Starting index of training set
   nret = 0 ;             // Number of computed returns


/*
   Do walkforward
*/

   for (;;) {

      // Train
      crit = opt_params ( which_crit ,  all_bars , n_train , prices + train_start ,
                          max_lookback , &lookback , &thresh , &last_pos ) ;
      printf ( "\n IS at %d  Lookback=%d  Thresh=%.3lf  Crit=%.3lf",
               train_start, lookback, thresh, mult * crit ) ;

      n = n_test ;    // Test this many cases
      if (n > nprices - train_start - n_train) // Don't go past the end of history
         n = nprices - train_start - n_train ;

      // Test
      comp_return ( ret_type , nprices , prices , train_start + n_train , n , lookback ,
                    thresh , last_pos , &n_returns , returns + nret ) ;
      nret += n_returns ;

      printf ( "\nOOS testing %d from %d had %d returns, total=%d",
               n, train_start + n_train, n_returns, nret ) ;

      // Advance fold window; quit if done
      train_start += n ;
      if (train_start + n_train >= nprices)
         break ;
      }

/*
   Compute and print OOS performance
*/

   printf ( "\n\nnprices=%d  max_lookback=%d  which_crit=%d  all_bars=%d  ret_type=%d  n_train=%d  n_test=%d",
            nprices, max_lookback, which_crit, all_bars, ret_type, n_train, n_test ) ;

   if (which_crit == 0) {
      crit = 0.0 ;
      for (i=0 ; i<nret ; i++)
         crit += returns[i] ;
      crit /= (nret + 1.e-60) ;
      printf ( "\n\nOOS mean return per open-trade bar (times 25200) = %.5lf  nret=%d", 25200 * crit, nret ) ;
      }

   else if (which_crit == 1) {
      win_sum = lose_sum = 1.e-60 ;
      for (i=0 ; i<nret ; i++) {
         if (returns[i] > 0.0)
            win_sum += returns[i] ;
         else if (returns[i] < 0.0)
            lose_sum -= returns[i] ;
         }
      crit = win_sum / lose_sum ;
      printf ( "\n\nOOS profit factor = %.5lf  nret=%d", crit, nret ) ;
      }

   else if (which_crit == 2) {
      sum = sum_squares = 0.0 ;
      for (i=0 ; i<nret ; i++) {
         sum += returns[i] ;
         sum_squares += returns[i] * returns[i] ;
         }
      sum /= (nret + 1.e-60) ;
      sum_squares /= (nret + 1.e-60) ;
      sum_squares -= sum * sum ;  // Variance (may be zero!)
      if (sum_squares < 1.e-20)   // Must not divide by zero or take sqrt of negative
         sum_squares = 1.e-20 ;
      crit = sum / sqrt ( sum_squares ) ;
      printf ( "\n\nOOS raw Sharpe ratio = %.5lf  nret=%d", crit, nret ) ;
      }
      

   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   free ( prices ) ;
   free ( returns ) ;

   exit ( 0 ) ;
}
