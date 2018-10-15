/******************************************************************************/
/*                                                                            */
/*  ENTROPY - Test entropy of indicators                                      */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>
#include <malloc.h>


#define MKTBUF 2048   /* Alloc for market info in chunks of this many records */
                      /* This is not critical and can be any reasonable vlaue */

/*
--------------------------------------------------------------------------------

   Local routine sorts ascending

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

   Local routine cleans tails

--------------------------------------------------------------------------------
*/


void clean_tails (
   int n ,            // Number of cases
   double *raw ,      // They are here
   double *work ,     // Work area n long
   double tail_frac   // Fraction of each tail to be cleaned (0-0.5)
   )
{
   int i, istart, istop, best_start, best_stop ;
   double cover, range, best, limit, scale, minval, maxval ;

   cover = 1.0 - 2.0 * tail_frac ;  // Internal fraction preserved

/*
   Find the interval having desired interior coverage which has the minimum data span
   Save the raw data, as we have to sort it for this step.
*/

   for (i=0 ; i<n ; i++)
      work[i] = raw[i] ;

   qsortd ( 0 , n-1 , work ) ;
   istart = 0 ;                        // Start search at the beginning
   istop = (int) (cover * (n+1)) - 1 ; // This gives desired coverage
   if (istop >= n)                     // Happens if careless user has tail=0
      istop = n - 1 ;

   best = 1.e60 ;                // Will be minimum span
   best_start = best_stop = 0 ;  // Not needed; shuts up LINT

   while (istop < n) {                    // Test every possible position
      range = work[istop] - work[istart] ; // This is what we minimize
      if (range < best) {
         best = range ;
         best_start = istart ;
         best_stop = istop ;
         }
      ++istart ;
      ++istop ;
      }

   minval = work[best_start] ;  // Value at start of interior interval
   maxval = work[best_stop] ;   // And end
   if (maxval <= minval) {      // Rare pathological situation
      maxval *= 1.0 + 1.e-10 ;
      minval *= 1.0 - 1.e-10 ;
      }

/*
   We now have the narrowest coverage.  Clean the tails.
   We use maxval-minval to keep reasonable scaling.
*/

   limit = (maxval - minval) * (1.0 - cover) ;
   scale = -1.0 / (maxval - minval) ;

   for (i=0 ; i<n ; i++) {
      if (raw[i] < minval)         // Left tail
         raw[i] = minval - limit * (1.0 - exp ( scale * (minval - raw[i]) ) ) ;
      else if (raw[i] > maxval)   // Right tail
         raw[i] = maxval + limit * (1.0 - exp ( scale * (raw[i] - maxval) ) ) ;
      }
}


/*
--------------------------------------------------------------------------------

   Local routine computes linear slope (trend)

--------------------------------------------------------------------------------
*/

double find_slope (
   int lookback ,   // Window length for computing slope indicator
   double *x        // Pointer to current price
   )
{
   int i ;
   double *pptr, coef, slope, denom ;

   pptr = x - lookback + 1 ;     // Indicator lookback window starts here
   slope = 0.0 ;                 // Will sum slope here
   denom = 0.0 ;                 // Will sum normalizer here

   for (i=0 ; i<lookback ; i++) {
      coef = i - 0.5 * (lookback - 1) ;
      denom += coef * coef ;
      slope += coef * *pptr++ ;
      }

   return slope / denom ;
}


/*
--------------------------------------------------------------------------------

   Local routine computes average true range

--------------------------------------------------------------------------------
*/

double atr (
   int lookback ,   // Window length for computing atr indicator
   double *high ,   // Pointer to current log high
   double *low ,    // Pointer to current log low
   double *close    // Pointer to current log close
   )
{
   int i ;
   double term, sum ;

   high -= lookback-1 ;   // Point to the first bar that will be used
   low -= lookback-1 ;
   close -= lookback-1 ;

   sum = 0.0 ;
   for (i=0 ; i<lookback ; i++) {
      term = high[i] - low[i] ;
      if (i) {     // Is there a prior bar?
         if (high[i] - close[i-1] > term)
            term = high[i] - close[i-1] ;
         if (close[i-1] - low[i] > term)
            term = close[i-1] - low[i] ;
         }
      sum += term ;
      }

   return sum / lookback ;
}


/*
--------------------------------------------------------------------------------

   Local routine computes range expansion
   This is a BAD indicator for demonstration only!

--------------------------------------------------------------------------------
*/

double range_expansion (
   int lookback ,   // Window length for computing expansion indicator
   double *x        // Pointer to current price
   )
{
   int i ;
   double *pptr, recent_high, recent_low, older_high, older_low ;

   pptr = x - lookback + 1 ; // Indicator lookback window starts here
   recent_high = older_high = -1.e60 ;
   recent_low = older_low = 1.e60 ;

   for (i=0 ; i<lookback/2 ; i++) {
      if (*pptr > older_high)
         older_high = *pptr ;
      if (*pptr < older_low)
         older_low = *pptr ;
      ++pptr ;
      }

   while (i++<lookback) {
      if (*pptr > recent_high)
         recent_high = *pptr ;
      if (*pptr < recent_low)
         recent_low = *pptr ;
      ++pptr ;
      }

   return (recent_high - recent_low) / (older_high - older_low + 1.e-10) ;
}


/*
----------------------------------------------------------------------------------

   Local routine computes price jump, optionally cleaned

   The exponential smoothing used here is not strictly correct, because it looks
   back over a limited window rather than all the way back to the beginning of
   the series.  But this is pretty close, and certainly good enough for this task.

----------------------------------------------------------------------------------
*/

double jump (
   int lookback ,     // Window length for computing indicator, greater than 1
   double *x          // Pointer to current price
   )
{
   int i ;
   double *pptr, alpha, smoothed ;

   alpha = 2.0 / lookback ; // Alpha = 2.0 / (n+1) and n=lookback-1 here
   pptr = x - lookback + 1 ; // Indicator lookback window starts here
   smoothed = *pptr++ ;

   for (i=1 ; i<lookback-1 ; i++)
      smoothed = alpha * *pptr++ + (1.0 - alpha) * smoothed ;

   return *pptr - smoothed ;
}


/*
--------------------------------------------------------------------------------

   Local routine computes relative entropy

--------------------------------------------------------------------------------
*/

double entropy (
   int n ,      // Number of data values
   double *x ,  // They are here
   int nbins ,  // Number of bins, at least 2
   int *count   // Work area nbins long
   )
{
   int i, k ;
   double minval, maxval, factor, p, sum ;

   minval = maxval = x[0] ;
   for (i=1 ; i<n ; i++) {
      if (x[i] < minval)
         minval = x[i] ;
      if (x[i] > maxval)
         maxval = x[i] ;
      }

   factor = (nbins - 1.e-10) / (maxval - minval + 1.e-60) ;

   for (i=0 ; i<nbins ; i++)
      count[i] = 0 ;

   for (i=0 ; i<n ; i++) {
      k = (int) (factor * (x[i] - minval)) ;
      ++count[k] ;
      }

   sum = 0.0 ;
   for (i=0 ; i<nbins ; i++) {
      if (count[i]) {
         p = (double) count[i] / n ;
         sum += p * log ( p ) ;
         }
      }

   return -sum / log ( (double) nbins ) ;
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
   int i, k, nprices, nind, lookback, bufcnt, *date, itemp, full_date, prior_date, year, month, day ;
   int nbins, *count, version, full_lookback ;
   double *open, *high, *low, *close, *trend, *volatility, *expansion, *raw_jump, *cleaned_jump, *work ;
   double trend_min, trend_max, volatility_min, volatility_max, expansion_min, expansion_max ;
   double raw_jump_min, raw_jump_max, cleaned_jump_min, cleaned_jump_max ;
   double trend_median, volatility_median, expansion_median, raw_jump_median, cleaned_jump_median ;
   double trend_entropy, volatility_entropy, expansion_entropy, raw_jump_entropy, cleaned_jump_entropy ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 5) {
      printf ( "\nUsage: ENTROPY  Lookback  Nbins  Version  Filename" ) ;
      printf ( "\n  lookback - Lookback for indicators" ) ;
      printf ( "\n  Nbins - Number of bins for entropy calculation" ) ;
      printf ( "\n  Version - 0=raw stat; 1=current-prior; >1=current-longer" ) ;
      printf ( "\n  Filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   lookback = atoi ( argv[1] ) ;
   nbins = atoi ( argv[2] ) ;
   version = atoi ( argv[3] ) ;
   strcpy_s ( filename , argv[4] ) ;
#else
   lookback = 20 ;
   nbins = 20 ;
   version = 0 ;
   strcpy_s ( filename , "E:\\MarketDataAssorted\\INDEXES\\SP-500.TXT" ) ;
#endif

/*
   Initialize
*/

   if (lookback < 2) {
      printf ( "\n\nLookback must be at least 2" ) ;
      exit ( 1 ) ;
      }

   if (version == 0)
      full_lookback = lookback ;
   else if (version == 1)
      full_lookback = 2 * lookback ;
   else if (version > 1)
      full_lookback = version * lookback ;
   else {
      printf ( "\n\nVersion cannot be negative" ) ;
      exit ( 1 ) ;
      }

   fp = NULL ;
   date = NULL ;
   open = NULL ;
   high = NULL ;
   low = NULL ;
   close = NULL ;
   trend = NULL ;
   volatility = NULL ;
   expansion = NULL ;
   count = NULL ;

/*
   Read market prices
*/

   if (fopen_s ( &fp, filename , "rt" )) {
      printf ( "\n\nCannot open market history file %s", filename ) ;
      exit ( 1 ) ;
      }

   date = (int *) malloc ( MKTBUF * sizeof(int) ) ;
   open = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   high = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   low = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   close = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   if (date == NULL  ||  open == NULL  ||  high == NULL  ||  low == NULL  ||  close == NULL) {
      printf ( "\n\nInsufficient memory reading market history file %s  Press any key...", filename ) ;
      goto FINISH ;
      }

   bufcnt = MKTBUF ;  // There are this many record slots available now

   printf ( "\nReading market file..." ) ;

   nprices = 0 ;    // Counts lines (prices) read
   prior_date = 0 ;

   for (;;) {

      if (feof ( fp )                          // If end of file
       || (fgets ( line , 256 , fp ) == NULL)  // Or unable to read line
       || (strlen ( line ) < 2))               // Or empty line
         break ;                               // We are done reading price history

      if (ferror ( fp )) {                     // If an error reading file
         printf ( "\nError reading line %d of file %s", nprices+1, filename ) ;
         goto FINISH ;
         }

      if (! bufcnt) {  // Allocate a new memory block if needed
         date = (int *) realloc ( date , (nprices+MKTBUF) * sizeof(int) ) ;
         open = (double *) realloc ( open , (nprices+MKTBUF) * sizeof(double) ) ;
         high = (double *) realloc ( high , (nprices+MKTBUF) * sizeof(double) ) ;
         low = (double *) realloc ( low , (nprices+MKTBUF) * sizeof(double) ) ;
         close = (double *) realloc ( close , (nprices+MKTBUF) * sizeof(double) ) ;
         if (date == NULL  ||  open == NULL  ||  high == NULL  ||  low == NULL  ||  close == NULL) {
            printf ( "\n\nInsufficient memory reading market history file %s  Press any key...", filename ) ;
            goto FINISH ;
            } // If insufficient memory
         bufcnt = MKTBUF ;  // There are this many new record slots available now
         } // If allocating new block

      // Parse the date and do a crude sanity check

      for (i=0 ; i<8 ; i++) {
         if ((line[i] < '0')  ||  (line[i] > '9')) {
            printf ( "\nInvalid date reading line %d of file %s", nprices+1, filename ) ;
            goto FINISH ;
            }
         }

      full_date = itemp = atoi ( line ) ;
      year = itemp / 10000 ;
      itemp -= year * 10000 ;
      month = itemp / 100 ;
      itemp -= month * 100 ;
      day = itemp ;

      if (month < 1 || month > 12 || day < 1 || day > 31 || year < 1800 || year > 2030) {
         printf ( "\nERROR... Invalid date %d in line %d", full_date, nprices+1 ) ;
         goto FINISH ;
         }

      if (full_date <= prior_date) {
         printf ( "\nERROR... Date failed to increase in line %d", nprices+1 ) ;
         goto FINISH ;
         }

      prior_date = full_date ;

      date[nprices] = full_date ;

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
         printf ( "\nInvalid open/high/low/close reading line %d of file %s", nprices+1, filename ) ;
         goto FINISH ;
         }

      ++nprices  ;
      --bufcnt ;           // One less slot remains

      } // For all lines

   fclose ( fp ) ;
   fp = NULL ;

   printf ( "\nMarket price history read (%d lines)", nprices ) ;
   printf ( "\n\nIndicator version %d", version ) ;

   count = (int *) malloc ( nbins * sizeof(int) ) ;
   if (count == NULL) {
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      goto FINISH ;
      }

/*
----------------------------------------
   The market is read.  Do computations.
----------------------------------------
*/

/*
   Trend
*/

   nind = nprices - full_lookback + 1 ;   // This many indicators

   trend = (double *) malloc ( nind * sizeof(double) ) ;
   if (trend == NULL) {
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      goto FINISH ;
      }

   trend_min = 1.e60 ;
   trend_max = -1.e60 ;
   for (i=0 ; i<nind ; i++) {
      k = full_lookback - 1 + i ;
      if (version == 0)
         trend[i] = find_slope ( lookback , close + k ) ;
      else if (version == 1)
         trend[i] = find_slope ( lookback , close + k ) - find_slope ( lookback , close + k - lookback ) ;
      else
         trend[i] = find_slope ( lookback , close + k ) - find_slope ( full_lookback , close + k ) ;
      if (trend[i] < trend_min)
         trend_min = trend[i] ;
      if (trend[i] > trend_max)
         trend_max = trend[i] ;
      }

   trend_entropy = entropy ( nind , trend , nbins , count ) ;

   qsortd ( 0 , nind-1 , trend ) ;
   if (nind % 2)
      trend_median = trend[nind/2] ;
   else 
      trend_median = 0.5 * (trend[nind/2-1] + trend[nind/2]) ;

   printf ( "\n\nTrend  min=%.4lf  max=%.4lf  median=%.4lf  relative entropy=%.3lf",
            trend_min, trend_max, trend_median, trend_entropy ) ;

/*
   Volatility
*/

   volatility = (double *) malloc ( nind * sizeof(double) ) ;
   if (volatility == NULL) {
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      goto FINISH ;
      }

   volatility_min = 1.e60 ;
   volatility_max = -1.e60 ;
   for (i=0 ; i<nind ; i++) {
      k = full_lookback - 1 + i ;
      if (version == 0)
         volatility[i] = atr ( lookback , high + k , low + k , close + k ) ;
      else if (version == 1)
         volatility[i] = atr ( lookback , high + k , low + k , close + k ) - atr ( lookback , high + k - lookback , low + k - lookback , close + k - lookback ) ;
      else
         volatility[i] = atr ( lookback , high + k , low + k , close + k ) - atr ( full_lookback , high + k , low + k , close + k ) ;
      if (volatility[i] < volatility_min)
         volatility_min = volatility[i] ;
      if (volatility[i] > volatility_max)
         volatility_max = volatility[i] ;
      }

   volatility_entropy = entropy ( nind , volatility , nbins , count ) ;

   qsortd ( 0 , nind-1 , volatility ) ;
   if (nind % 2)
      volatility_median = volatility[nind/2] ;
   else 
      volatility_median = 0.5 * (volatility[nind/2-1] + volatility[nind/2]) ;

   printf ( "\n\nVolatility  min=%.4lf  max=%.4lf  median=%.4lf   relative entropy=%.3lf",
            volatility_min, volatility_max, volatility_median, volatility_entropy ) ;

/*
   Expansion
*/

   expansion = (double *) malloc ( nind * sizeof(double) ) ;
   if (expansion == NULL) {
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      goto FINISH ;
      }

   expansion_min = 1.e60 ;
   expansion_max = -1.e60 ;
   for (i=0 ; i<nind ; i++) {
      k = full_lookback - 1 + i ;
      if (version == 0)
         expansion[i] = range_expansion ( lookback , close + k ) ;
      else if (version == 1)
         expansion[i] = range_expansion ( lookback , close + k ) - range_expansion ( lookback , close + k - lookback ) ;
      else
         expansion[i] = range_expansion ( lookback , close + k ) - range_expansion ( full_lookback , close + k ) ;
      if (expansion[i] < expansion_min)
         expansion_min = expansion[i] ;
      if (expansion[i] > expansion_max)
         expansion_max = expansion[i] ;
      }

   expansion_entropy = entropy ( nind , expansion , nbins , count ) ;

   qsortd ( 0 , nind-1 , expansion ) ;
   if (nind % 2)
      expansion_median = expansion[nind/2] ;
   else 
      expansion_median = 0.5 * (expansion[nind/2-1] + expansion[nind/2]) ;

   printf ( "\n\nExpansion  min=%.4lf  max=%.4lf  median=%.4lf   relative entropy=%.3lf",
            expansion_min, expansion_max, expansion_median, expansion_entropy ) ;

/*
   Raw jump
*/

   raw_jump = (double *) malloc ( nind * sizeof(double) ) ;
   if (raw_jump == NULL) {
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      goto FINISH ;
      }

   raw_jump_min = 1.e60 ;
   raw_jump_max = -1.e60 ;
   for (i=0 ; i<nind ; i++) {
      k = full_lookback - 1 + i ;
      if (version == 0)
         raw_jump[i] = jump ( lookback , close + k ) ;
      else if (version == 1)
         raw_jump[i] = jump ( lookback , close + k ) - jump ( lookback , close + k - lookback ) ;
      else
         raw_jump[i] = jump ( lookback , close + k ) - jump ( full_lookback , close + k ) ;
      if (raw_jump[i] < raw_jump_min)
         raw_jump_min = raw_jump[i] ;
      if (raw_jump[i] > raw_jump_max)
         raw_jump_max = raw_jump[i] ;
      }

   raw_jump_entropy = entropy ( nind , raw_jump , nbins , count ) ;

   qsortd ( 0 , nind-1 , raw_jump ) ;
   if (nind % 2)
      raw_jump_median = raw_jump[nind/2] ;
   else 
      raw_jump_median = 0.5 * (raw_jump[nind/2-1] + raw_jump[nind/2]) ;

   printf ( "\n\nRawJump  min=%.4lf  max=%.4lf  median=%.4lf  relative entropy=%.3lf",
            raw_jump_min, raw_jump_max, raw_jump_median, raw_jump_entropy ) ;

/*
   Cleaned jump
*/

   cleaned_jump = (double *) malloc ( 2 * nind * sizeof(double) ) ;
   if (cleaned_jump == NULL) {
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      goto FINISH ;
      }
   work = cleaned_jump + nind ;

   for (i=0 ; i<nind ; i++)
      cleaned_jump[i] = raw_jump[i] ;

   clean_tails ( nind , cleaned_jump , work , 0.05 ) ;

   cleaned_jump_min = 1.e60 ;
   cleaned_jump_max = -1.e60 ;
   for (i=0 ; i<nind ; i++) {
      if (cleaned_jump[i] < cleaned_jump_min)
         cleaned_jump_min = cleaned_jump[i] ;
      if (cleaned_jump[i] > cleaned_jump_max)
         cleaned_jump_max = cleaned_jump[i] ;
      }

   cleaned_jump_entropy = entropy ( nind , cleaned_jump , nbins , count ) ;

   qsortd ( 0 , nind-1 , cleaned_jump ) ;
   if (nind % 2)
      cleaned_jump_median = cleaned_jump[nind/2] ;
   else 
      cleaned_jump_median = 0.5 * (cleaned_jump[nind/2-1] + cleaned_jump[nind/2]) ;

   printf ( "\n\nCleanedJump  min=%.4lf  max=%.4lf  median=%.4lf  relative entropy=%.3lf",
            cleaned_jump_min, cleaned_jump_max, cleaned_jump_median, cleaned_jump_entropy ) ;

#if 0
   _getch() ;
   for (i=0 ; i<nind ; i++)
      printf ( "\n%12.5lf %12.5lf", raw_jump[i], cleaned_jump[i] ) ;
   _getch() ;
#endif

FINISH:
   if (_heapchk() != _HEAPOK) {
      printf ( "\nBad heap!" ) ;
      exit ( 1 ) ;
      }

   if (fp != NULL)
      fclose ( fp ) ;
   if (date != NULL)
      free ( date ) ;
   if (open != NULL)
      free ( open ) ;
   if (high != NULL)
      free ( high ) ;
   if (low != NULL)
      free ( low ) ;
   if (close != NULL)
      free ( close ) ;
   if (trend != NULL)
      free ( trend ) ;
   if (volatility != NULL)
      free ( volatility ) ;
   if (expansion != NULL)
      free ( expansion ) ;
   if (count != NULL)
      free ( count ) ;
   if (raw_jump != NULL)
      free ( raw_jump ) ;
   if (cleaned_jump != NULL)
      free ( cleaned_jump ) ;

   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   exit ( 0 ) ;
}
