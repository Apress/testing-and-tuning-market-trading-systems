/******************************************************************************/
/*                                                                            */
/* MCPT_BARS - Use a mean-reversion system with bar data to demonstrate       */
/*             two Monte-Carlo permutation test techniques:                   */
/*             1) Test for outstanding performance with unpermuted data       */
/*                This simultaneously tests for excessive weakness and        */
/*                excessive strength (overfitting)                            */
/*             2) Estimate true skill and unbiased future return              */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <malloc.h>
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

   Local routine computes optimal long-term rise and short-term drop
   for a primitive mean reversion long-only system
   This uses the more conservative next-open-to-open return.

--------------------------------------------------------------------------------
*/

double opt_params (   // Returns total log profit starting at lookback
   int ncases ,       // Number of log prices
   int lookback ,     // Lookback for long-term rise
   double *open ,     // Log of open prices
   double *close ,    // Log of close prices
   double *opt_rise , // Returns optimal long-term rise threshold
   double *opt_drop , // Returns optimal short-term drop threshold
   int *nlong         // Number of long returns
   )
{
   int i, irise, idrop, nl ;
   double total_return, best_perf, ret, rise, drop, rise_thresh, drop_thresh ;

   best_perf = -1.e60 ;                       // Will be best performance across all trials
   for (irise=1 ; irise<=50 ; irise++) {      // Trial long-term rise
      rise_thresh = irise * 0.005 ;
      for (idrop=1 ; idrop<=50 ; idrop++) {   // Trial short-term drop
         drop_thresh = idrop * .0005 ;

         // We have a pair of thresholds to try.
         // Cumulate performance for all valid cases.
         // Start at lookback.

         total_return = 0.0 ;      // Cumulate total return for this trial
         nl = 0 ;                  // Will count long positions

         for (i=lookback ; i<ncases-2 ; i++) { // Compute performance across history

            rise = close[i] - close[i-lookback] ;
            drop = close[i-1] - close[i] ;

            if (rise >= rise_thresh  &&  drop >= drop_thresh) {
               ret = open[i+2] - open[i+1] ;
               ++nl ;
               }
            else
               ret = 0.0 ;

            total_return += ret ;
            } // For i, summing performance for this trial

         // We now have the performance figures across the history
         // Keep track of the best

         if (total_return > best_perf) {  // Did this trial param set break a record?
            best_perf = total_return ;
            *opt_rise = rise_thresh ;
            *opt_drop = drop_thresh ;
            *nlong = nl ;
            }

         } // For idrop
      } // For irise

   return best_perf ;
}


/*
--------------------------------------------------------------------------------

   Permutation routines (assumes prices are actually log prices)

   We have nc cases from index 0 through nc-1.
   The first case is the 'basis' case and remains unchanged.
   The last case examined is at nc-1, but its close will remain unchanged.
   Thus, the shuffled array starts and ends at their original values.
   Only the interior elements change.

   If preserve_OO is nonzero, then the first close-to-open change and the
   last open-to-close change are not permuted.  This preserves the
   open-to-open price difference at the ends.
   This is appropriate for next-open-to-open return scoring.

--------------------------------------------------------------------------------
*/

void prepare_permute (
   int nc ,            // Number of bars
   double *open ,      // Input of nc log prices
   double *high ,
   double *low ,
   double *close ,
   double *rel_open ,  // Work area; returns computed changes
   double *rel_high ,
   double *rel_low ,
   double *rel_close
   )
{
   int icase ;

   for (icase=1 ; icase<nc ; icase++) {
      rel_open[icase-1] = open[icase] - close[icase-1] ;
      rel_high[icase-1] = high[icase] - open[icase] ;
      rel_low[icase-1] = low[icase] - open[icase] ;
      rel_close[icase-1] = close[icase] - open[icase] ;
      }
}

void do_permute (
   int nc ,          // Number of cases
   int preserve_OO , // Preserve next open-to-open (vs first open to last close)
   double *open ,    // Returns nc shuffled log prices
   double *high ,
   double *low ,
   double *close ,
   double *rel_open ,  // Work area; input of computed changes
   double *rel_high ,
   double *rel_low ,
   double *rel_close
   )
{
   int i, j, icase ;
   double dtemp ;

   if (preserve_OO)
      preserve_OO = 1 ;

   // Shuffle the close-to-open changes.

   i = nc-1-preserve_OO ; // Number remaining to be shuffled
   while (i > 1) {        // While at least 2 left to shuffle
      j = (int) (unifrand() * i) ;
      if (j >= i)         // Should never happen, but be safe
         j = i - 1 ;
      --i ;
      dtemp = rel_open[i+preserve_OO] ;
      rel_open[i+preserve_OO] = rel_open[j+preserve_OO] ;
      rel_open[j+preserve_OO] = dtemp ;
      } // Shuffle the changes

   // Shuffle the open-to-close changes.

   i = nc-1-preserve_OO ; // Number remaining to be shuffled
   while (i > 1) {        // While at least 2 left to shuffle
      j = (int) (unifrand() * i) ;
      if (j >= i)         // Should never happen, but be safe
         j = i - 1 ;
      --i ;
      dtemp = rel_high[i] ;
      rel_high[i] = rel_high[j] ;
      rel_high[j] = dtemp ;
      dtemp = rel_low[i] ;
      rel_low[i] = rel_low[j] ;
      rel_low[j] = dtemp ;
      dtemp = rel_close[i] ;
      rel_close[i] = rel_close[j] ;
      rel_close[j] = dtemp ;
      } // Shuffle the changes

   // Now rebuild the prices, using the shuffled changes

   for (icase=1 ; icase<nc ; icase++) {
      open[icase] = close[icase-1] + rel_open[icase-1] ;
      high[icase] = open[icase] + rel_high[icase-1] ;
      low[icase] = open[icase] + rel_low[icase-1] ;
      close[icase] = open[icase] + rel_close[icase-1] ;
      }
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
   int i, irep, nreps, nprices, bufcnt, lookback, count ;
   int nlong, original_nlong ;
   double *open, *high, *low, *close, *rel_open, *rel_high, *rel_low, *rel_close, opt_return, original, opt_rise, opt_drop ;
   double trend_per_return, trend_component, original_trend_component, training_bias, mean_training_bias, unbiased_return, skill ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 4) {
      printf ( "\nUsage: MCPT_BARS  lookback  nreps  filename" ) ;
      printf ( "\n  lookback - Long-term rise lookback" ) ;
      printf ( "\n  nreps - Number of MCPT replications (hundreds or thousands)" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Open High Low Close)" ) ;
      exit ( 1 ) ;
      }

   lookback = atoi ( argv[1] ) ;
   nreps = atoi ( argv[2] ) ;
   strcpy_s ( filename , argv[3] ) ;
#else
   lookback = 300 ;
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

   open = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   high = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   low = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   close = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   if (open == NULL  ||  high == NULL  ||  low == NULL  ||  close == NULL) {
      if (open != NULL)
         free ( open ) ;
      if (high != NULL)
         free ( high ) ;
      if (low != NULL)
         free ( low ) ;
      if (close != NULL)
         free ( close ) ;
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
         free ( open ) ;
         free ( high ) ;
         free ( low ) ;
         free ( close ) ;
         printf ( "\nError reading line %d of file %s", nprices+1, filename ) ;
         exit ( 1 ) ;
         }

      if (! bufcnt) {  // Allocate a new memory block if needed
         open = (double *) realloc ( open , (nprices+MKTBUF) * sizeof(double) ) ;
         high = (double *) realloc ( high , (nprices+MKTBUF) * sizeof(double) ) ;
         low = (double *) realloc ( low , (nprices+MKTBUF) * sizeof(double) ) ;
         close = (double *) realloc ( close , (nprices+MKTBUF) * sizeof(double) ) ;
         if (open == NULL  ||  high == NULL  ||  low == NULL  ||  close == NULL) {
            if (open != NULL)
               free ( open ) ;
            if (high != NULL)
               free ( high ) ;
            if (low != NULL)
               free ( low ) ;
            if (close != NULL)
               free ( close ) ;
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
            free ( open ) ;
            free ( high ) ;
            free ( low ) ;
            free ( close ) ;
            printf ( "\nInvalid date reading line %d of file %s", nprices+1, filename ) ;
            exit ( 1 ) ;
            }
         }

      // Parse the open

      cptr = line + 9 ;  // Price is in this column or beyond
                         // (Next loop allows price to start past this)

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      open[nprices] = atof ( cptr ) ;
      if (open[nprices] > 0.0)                     // Always true, but avoid disaster
         open[nprices] = log ( open[nprices] ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the high

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      high[nprices] = atof ( cptr ) ;
      if (high[nprices] > 0.0)                     // Always true, but avoid disaster
         high[nprices] = log ( high[nprices] ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the low

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      low[nprices] = atof ( cptr ) ;
      if (low[nprices] > 0.0)                     // Always true, but avoid disaster
         low[nprices] = log ( low[nprices] ) ;

      while (*cptr  &&  *cptr != ' '  &&  *cptr != ','  &&  *cptr != '\t')
         ++cptr ;  // Pass the price; stop at delimiter

      // Parse the close

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      close[nprices] = atof ( cptr ) ;
      if (close[nprices] > 0.0)                     // Always true, but avoid disaster
         close[nprices] = log ( close[nprices] ) ;

      if (low[nprices] > open[nprices]  ||  low[nprices] > close[nprices]  ||
          high[nprices] < open[nprices]  ||  high[nprices] < close[nprices]) {
         fclose ( fp ) ;
         free ( open ) ;
         free ( high ) ;
         free ( low ) ;
         free ( close ) ;
         printf ( "\nInvalid open/high/low/close reading line %d of file %s", nprices+1, filename ) ;
         exit ( 1 ) ;
         }

      ++nprices  ;
      --bufcnt ;           // One less slot remains

      } // For all lines

   fclose ( fp ) ;

   printf ( "\nMarket price history read" ) ;


/*
   The market data is read.  Initialize for MCPT.
   Evaluation period starts at lookback.
*/

   if (nprices - lookback < 10) {
      printf ( "\nERROR... Number of prices must be at least 10 greater than lookback" ) ;
      exit ( 1 ) ;
      }

   rel_open = (double *) malloc ( 4 * nprices * sizeof(double) ) ;
   if (rel_open == NULL) {
      printf ( "\n\nInsufficient memory.   Press any key..." ) ;
      free ( open ) ;
      free ( high ) ;
      free ( low ) ;
      free ( close ) ;
      _getch () ;  // Wait for user to press a key
      exit ( 1 ) ;
      } // If insufficient memory

   rel_high = rel_open + nprices ;
   rel_low = rel_high + nprices ;
   rel_close = rel_low + nprices ;

   trend_per_return = (open[nprices-1] - open[lookback+1]) / (nprices - lookback - 2) ;

   prepare_permute ( nprices-lookback , open+lookback , high+lookback , low+lookback , close+lookback ,
                     rel_open , rel_high , rel_low , rel_close ) ;

/*
   Do MCPT
*/

   for (irep=0 ; irep<nreps ; irep++) {

      if (irep)   // Shuffle
         do_permute ( nprices-lookback , 1 , open+lookback , high+lookback , low+lookback , close+lookback ,
                      rel_open , rel_high , rel_low , rel_close ) ;

      opt_return = opt_params ( nprices , lookback , open , close , &opt_rise , &opt_drop , &nlong ) ;
      trend_component = nlong * trend_per_return ;
      printf ( "\n%5d: Ret = %.3lf  Rise, drop= %.4lf %.4lf  NL=%d  TrndComp=%.4lf  TrnBias=%.4lf",
               irep, opt_return, opt_rise, opt_drop, nlong, trend_component, opt_return - trend_component ) ;

      if (irep == 0) {
         original = opt_return ;
         original_trend_component = trend_component ;
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

   printf ( "\n\n%d prices were read, %d MCP replications with lookback = %d",
           nprices, nreps, lookback ) ;
   printf ( "\n\np-value for null hypothesis that system is worthless = %.4lf", (double) count / (double) nreps ) ;
   printf ( "\nTotal trend = %.4lf", open[nprices-1] - open[lookback+1] ) ;
   printf ( "\nOriginal nlong = %d", original_nlong ) ;
   printf ( "\nOriginal return = %.4lf", original ) ;
   printf ( "\nTrend component = %.4lf", original_trend_component ) ;
   printf ( "\nTraining bias = %.4lf", mean_training_bias ) ;
   printf ( "\nSkill = %.4lf", skill ) ;
   printf ( "\nUnbiased return = %.4lf", unbiased_return ) ;


   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   free ( open ) ;
   free ( high ) ;
   free ( low ) ;
   free ( close ) ;
   free ( rel_open ) ;

   exit ( 0 ) ;
}
