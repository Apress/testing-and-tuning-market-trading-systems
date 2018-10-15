/*****************************************************************************/
/*                                                                           */
/*  DEV_MA - Train a thresholded moving-average-crossover system using       */
/*           differential evolution                                          */
/*                                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>
#include <malloc.h>
#include "headers.h"

#define MKTBUF 2048   /* Alloc for market info in chunks of this many records */
                      /* This is not critical and can be any reasonable value */

// These pass the price data to the criterion function
static int local_n ;
static int local_max_lookback ;
static double *local_prices ;
static StocBias *stoc_bias ;   // Use the StocBias class to generate a cheap and very rough
                               // estimate of training bias from the differential evolution
                               // initialization data

/*
--------------------------------------------------------------------------------

   Local routine evaluates a thresholded moving-average crossover system
   This computes the total return.  Users may wish to change it to
   compute other criteria.

--------------------------------------------------------------------------------
*/

double test_system (
   int ncases ,          // Number of prices in history
   int max_lookback ,    // Max lookback that will ever be used
   double *x ,           // Log prices
   int long_term ,       // Long-term lookback
   double short_pct ,    // Short-term lookback is this / 100 times long_term, 0-100
   double short_thresh , // Short threshold times 10000
   double long_thresh ,  // Long threshold times 10000
   int *ntrades ,        // Returns number of trades
   double *returns       // If non-NULL returns ncases-max_lookback bar returns
   )
{
   int i, j, k, short_term ;
   double sum, ret, short_mean, long_mean, change ;

   short_term = (int) (0.01 * short_pct * long_term) ;
   if (short_term < 1)
      short_term = 1 ;
   if (short_term >= long_term)
      short_term = long_term - 1 ;
   short_thresh /= 10000.0 ;
   long_thresh /= 10000.0 ;

   sum = 0.0 ;                         // Cumulate performance for this trial
   *ntrades = 0 ;
   k = 0 ;                             // Will index returns

   for (i=max_lookback-1 ; i<ncases-1 ; i++) {   // Sum performance across history
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

      change = short_mean / long_mean - 1.0 ;  // Fractional difference in MA of log prices

      if (change > long_thresh) {        // Long position
         ret = x[i+1] - x[i] ;
         ++(*ntrades) ;
         }

      else if (change < -short_thresh) { // Short position
         ret = x[i] - x[i+1] ;
         ++(*ntrades) ;
         }

      else
         ret = 0.0 ;

      sum += ret ;

      if (returns != NULL)
         returns[k++] = ret ;

      } // For i, summing performance for this trial

   return sum ;
}


/*
--------------------------------------------------------------------------------

   This is the criterion function called from DIFF_EV.CPP.

--------------------------------------------------------------------------------
*/

double criter ( double *params , int mintrades )
{
   int long_term, ntrades ;
   double short_pct, short_thresh, long_thresh, ret_val ;

   long_term = (int) (params[0] + 1.e-10) ;
   short_pct = params[1] ;
   short_thresh = params[2] ;
   long_thresh = params[3] ;

   ret_val = test_system ( local_n , local_max_lookback , local_prices , long_term , short_pct ,
                           short_thresh , long_thresh , &ntrades ,
                           (stoc_bias != NULL) ? stoc_bias->expose_returns() : NULL ) ;

   if (stoc_bias != NULL  &&  ret_val > 0.0)
      stoc_bias->process () ;

   if (ntrades >= mintrades)
      return ret_val ;
   else
      return -1.e20 ;
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
   int i, nprices, bufcnt, max_lookback, ret_code, mintrades ;
   double *prices, max_thresh, low_bounds[4], high_bounds[4], params[5] ;
   char line[256], filename[4096], *cptr ;
   double IS_mean, OOS_mean, bias ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 4) {
      printf ( "\nUsage: DEV_MA  max_lookback  max_thresh  filename" ) ;
      printf ( "\n  max_lookback - Maximum moving-average lookback" ) ;
      printf ( "\n  max_thresh - Maximum fraction threshold times 10000" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   max_lookback = atoi ( argv[1] ) ;
   max_thresh = atof ( argv[2] ) ;
   strcpy_s ( filename , argv[3] ) ;
#else
   max_lookback = 100 ;
   max_thresh = 100.0 ;
   strcpy_s ( filename , "E:\\MarketDataAssorted\\INDEXES\\$OEX.TXT" ) ;
#endif

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

   printf ( "\nMarket price history read, %d prices", nprices ) ;


/*
   The market data is read.
*/

   local_n = nprices ;
   local_max_lookback = max_lookback ;
   local_prices = prices ;

   low_bounds[0] = 2 ;
   low_bounds[1] = 0.01 ;
   low_bounds[2] = 0.0 ;
   low_bounds[3] = 0.0 ;

   high_bounds[0] = max_lookback ;
   high_bounds[1] = 99.0 ;
   high_bounds[2] = max_thresh ;  // These are 10000 times actual threshold
   high_bounds[3] = max_thresh ;

   mintrades = 20 ;

   stoc_bias = new StocBias ( nprices - max_lookback ) ;   // This many returns
   if (stoc_bias == NULL  ||  ! stoc_bias->ok) {
      if (stoc_bias != NULL) {
         delete stoc_bias ;
         stoc_bias = NULL ;
         }
      free ( prices ) ;
      printf ( "\n\nInsufficient memory... Press any key..." ) ;
      _getch () ;  // Wait for user to press a key
      exit ( 1 ) ;
      }

/*
   Optimize and print best parameters and performance
*/

   ret_code = diff_ev ( criter , 4 , 1 , 100 , 10000 , mintrades , 10000000 , 300 , 0.2 , 0.2 , 0.3 , low_bounds , high_bounds , params , 1 , stoc_bias ) ;

   // Error returns should be handled here

   printf ( "\n\nBest performance = %.4lf  Variables follow...", params[4] ) ;
   for (i=0 ; i<4 ; i++)
      printf ( "\n  %.4lf", params[i] ) ;

/*
   Compute and print stochastic bias estimate
*/

   stoc_bias->compute ( &IS_mean , &OOS_mean , &bias ) ;

   printf ( "\n\nVery rough estimates from differential evolution initialization..." ) ;
   printf ( "\n  In-sample mean = %.4lf", IS_mean ) ;
   printf ( "\n  Out-of-sample mean = %.4lf", OOS_mean ) ;
   printf ( "\n  Bias = %.4lf", bias ) ;
   printf ( "\n  Expected = %.4lf", params[4] - bias ) ;

   delete stoc_bias ;
   stoc_bias = NULL ;  // Needed so criter() does not process returns in sensitivity()

/*
   Compute and print parameter sensitivity curves
*/

   ret_code = sensitivity ( criter , 4 , 1 , 30 , 80 , mintrades , params , low_bounds , high_bounds ) ;
   // handle error return here

   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   free ( prices ) ;
   exit ( 0 ) ;
}
