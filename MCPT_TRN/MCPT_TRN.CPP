/*****************************************************************************/
/*                                                                           */
/* MCPT_TRN - Use a primitive moving-average-crossover system to demonstrate */
/*            two Monte-Carlo permutation test techniques:                   */
/*            1) Test for outstanding performance with unpermuted data       */
/*               This simultaneously tests for excessive weakness and        */
/*               excessive strength (overfitting)                            */
/*            2) Estimate true skill and unbiased future return              */
/*                                                                           */
/*****************************************************************************/

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

   Local routine computes optimal short-term and long-term lookbacks
   for a primitive moving-average crossover system

--------------------------------------------------------------------------------
*/

double opt_params (   // Returns total log profit starting at max_lookback-1
   int ncases ,       // Number of log prices in X
   int max_lookback , // Maximum lookback to try
   double *x ,        // Log prices
   int *short_term ,  // Returns optimal short-term lookback
   int *long_term ,   // Returns optimal long-term lookback
   int *nshort ,      // Number of short returns
   int *nlong         // Number of long returns
   )
{
   int i, j, ishort, ilong, nl, ns ;
   double short_sum, long_sum, short_mean, long_mean, total_return, best_perf, ret ;

   best_perf = -1.e60 ;                            // Will be best performance across all trials
   for (ilong=2 ; ilong<=max_lookback ; ilong++) { // Trial long-term lookback
      for (ishort=1 ; ishort<ilong ; ishort++) {   // Trial short-term lookback

         // We have a pair of lookbacks to try.
         // Cumulate performance for all valid cases
         // Start at max_lookback-1 regardless of ilong for conformity

         total_return = 0.0 ;                    // Cumulate total return for this trial
         nl = ns = 0 ;                           // Will count long and short positions

         for (i=max_lookback-1 ; i<ncases-1 ; i++) {    // Compute performance across history

            if (i == max_lookback-1) {           // Find the short-term and long-term moving averages for first case.
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

            if (short_mean > long_mean) {      // Long position
               ret = x[i+1] - x[i] ;
               ++nl ;
               }
            else if (short_mean < long_mean) { // Short position
               ret = x[i] - x[i+1] ;
               ++ns ;
               }
            else
               ret = 0.0 ;

            total_return += ret ;
            } // For i, summing performance for this trial

         // We now have the performance figures across the history
         // Keep track of the best

         if (total_return > best_perf) {  // Did this trial param set break a record?
            best_perf = total_return ;
            *short_term = ishort ;
            *long_term = ilong ;
            *nlong = nl ;
            *nshort = ns ;
            }

         } // For ishort, all short-term lookbacks
      } // For ilong, all long-term lookbacks

   return best_perf ;
}


/*
--------------------------------------------------------------------------------

   Permutation routines (assumes prices in 'data' are actually log prices)

   We have nc cases from index 0 through nc-1.
   The first case is the 'basis' case and remains unchanged.
   The last case examined is at nc-1, but it, too, will remain unchanged.
   Thus, the shuffled array starts and ends at their original values.
   Only the interior elements change.

--------------------------------------------------------------------------------
*/

void prepare_permute (
   int nc ,           // Number of cases
   double *data ,     // Input of nc log prices
   double *changes    // Work area; returns computed changes
   )
{
   int icase ;

   for (icase=1 ; icase<nc ; icase++)
      changes[icase-1] = data[icase] - data[icase-1] ;
}

void do_permute (
   int nc ,          // Number of cases
   double *data ,    // Returns nc shuffled prices
   double *changes   // Work area; computed changes from prepare_permute
   )
{
   int i, j, icase ;
   double dtemp ;

   // Shuffle the changes. Do not include the first case in the shuffling,
   // as it is the starting price, not a change.

   i = nc-1 ;             // Number remaining to be shuffled
   while (i > 1) {        // While at least 2 left to shuffle
      j = (int) (unifrand() * i) ;
      if (j >= i)         // Should never happen, but be safe
         j = i - 1 ;
      --i ;
      dtemp = changes[i] ;
      changes[i] = changes[j] ;
      changes[j] = dtemp ;
      } // Shuffle the changes

   // Now rebuild the prices, using the shuffled changes

   for (icase=1 ; icase<nc ; icase++)
      data[icase] = data[icase-1] + changes[icase-1] ;
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
   int i, irep, nreps, nprices, bufcnt, max_lookback, long_lookback, short_lookback, count ;
   int nlong, nshort, original_nlong, original_nshort ;
   double *prices, *changes, opt_return, original ;
   double trend_per_return, trend_component, original_trend_component, training_bias, mean_training_bias, unbiased_return, skill ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 4) {
      printf ( "\nUsage: MCPT_TRN  max_lookback  nreps  filename" ) ;
      printf ( "\n  max_lookback - Maximum moving-average lookback" ) ;
      printf ( "\n  nreps - Number of MCPT replications (hundreds or thousands)" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   max_lookback = atoi ( argv[1] ) ;
   nreps = atoi ( argv[2] ) ;
   strcpy_s ( filename , argv[3] ) ;
#else
   max_lookback = 300 ;
   nreps = 10 ;
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

   printf ( "\nMarket price history read" ) ;


/*
   The market data is read.  Initialize for MCPT.
   For conformity, evaluation period starts at max_lookback-1.
*/

   if (nprices - max_lookback < 10) {
      printf ( "\nERROR... Number of prices must be at least 10 greater than max_lookback" ) ;
      exit ( 1 ) ;
      }

   changes = (double *) malloc ( nprices * sizeof(double) ) ;
   if (changes == NULL) {
      printf ( "\n\nInsufficient memory.   Press any key..." ) ;
      free ( prices ) ;
      _getch () ;  // Wait for user to press a key
      exit ( 1 ) ;
      } // If insufficient memory

   trend_per_return = (prices[nprices-1] - prices[max_lookback-1]) / (nprices - max_lookback) ;

   prepare_permute ( nprices-max_lookback+1 , prices+max_lookback-1 , changes ) ;

/*
   Do MCPT
*/

   for (irep=0 ; irep<nreps ; irep++) {

      if (irep)   // Shuffle
         do_permute ( nprices-max_lookback+1 , prices+max_lookback-1 , changes ) ;

      opt_return = opt_params ( nprices , max_lookback , prices , &short_lookback , &long_lookback , &nshort , &nlong ) ;
      trend_component = (nlong - nshort) * trend_per_return ;
      printf ( "\n%5d: Ret = %.3lf  Lookback=%d %d  NS, NL=%d %d  TrndComp=%.4lf  TrnBias=%.4lf",
               irep, opt_return, short_lookback, long_lookback, nshort, nlong, trend_component, opt_return - trend_component ) ;

      if (irep == 0) {
         original = opt_return ;
         original_trend_component = trend_component ;
         original_nshort = nshort ;
         original_nlong = nlong ;
         count = 1 ;
         mean_training_bias = 0.0 ;
         }

      else {
         training_bias = opt_return - trend_component ;
         mean_training_bias += training_bias ;
         if (opt_return >= original)
            ++count ;
         }
      }

   mean_training_bias /= (nreps - 1) ;
   unbiased_return = original - mean_training_bias ;
   skill = unbiased_return - original_trend_component ;

   printf ( "\n\n%d prices were read, %d MCP replications with max lookback = %d",
           nprices, nreps, max_lookback ) ;
   printf ( "\n\np-value for null hypothesis that system is worthless = %.4lf", (double) count / (double) nreps ) ;
   printf ( "\nTotal trend = %.4lf", prices[nprices-1] - prices[max_lookback-1] ) ;
   printf ( "\nOriginal nshort = %d", original_nshort ) ;
   printf ( "\nOriginal nlong = %d", original_nlong ) ;
   printf ( "\nOriginal return = %.4lf", original ) ;
   printf ( "\nTrend component = %.4lf", original_trend_component ) ;
   printf ( "\nTraining bias = %.4lf", mean_training_bias ) ;
   printf ( "\nSkill = %.4lf", skill ) ;
   printf ( "\nUnbiased return = %.4lf", unbiased_return ) ;


   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   free ( prices ) ;
   free ( changes ) ;
   exit ( 0 ) ;
}
