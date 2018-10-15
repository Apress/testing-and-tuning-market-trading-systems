/******************************************************************************/
/*                                                                            */
/*  STATN - Simple but effective stationarity test                            */
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

#define NGAPS 11      /* Number of gaps in analysis */

void qsortd ( int first , int last , double *data ) ;

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

   Local routine does gap analysis

--------------------------------------------------------------------------------
*/

void gap_analyze (
   int n ,
   double *x ,
   double thresh ,
   int ngaps ,
   int *gap_size ,
   int *gap_count
   )
{
   int i, j, above_below, new_above_below, count ;

   for (i=0 ; i<ngaps ; i++)
      gap_count[i] = 0 ;

   count = 1 ;
   above_below = (x[0] >= thresh)  ?  1 : 0 ;

   for (i=1 ; i<=n ; i++) {

      if (i == n)    // Passing end of array counts as a change
         new_above_below = 1 - above_below ;
      else
         new_above_below = (x[i] >= thresh)  ?  1 : 0 ;

      if (new_above_below == above_below)
         ++count ;
      else {
         for (j=0 ; j<ngaps-1 ; j++) {
            if (count <= gap_size[j])
               break ;
            }
         ++gap_count[j] ;
         count = 1 ;
         above_below = new_above_below ;
         }
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
   int i, k, nprices, nind, lookback, bufcnt, *date, itemp, full_date, prior_date, year, month, day ;
   int ngaps, gap_size[NGAPS-1], gap_count[NGAPS], version, full_lookback ;
   double *open, *high, *low, *close, *trend, *trend_sorted, *volatility, *volatility_sorted, fractile ;
   double trend_min, trend_max, trend_quantile, volatility_min, volatility_max, volatility_quantile ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 5) {
      printf ( "\nUsage: STATN  Lookback  Fractile  Version  Filename" ) ;
      printf ( "\n  lookback - Lookback for trend and volatility" ) ;
      printf ( "\n  fractile - Fractile (0-1, typically 0.5) for gap analysis" ) ;
      printf ( "\n  version - 0=raw stat; 1=current-prior; >1=current-longer" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   lookback = atoi ( argv[1] ) ;
   fractile = atof ( argv[2] ) ;
   version = atoi ( argv[3] ) ;
   strcpy_s ( filename , argv[4] ) ;
#else
   lookback = 100 ;
   fractile = 0.5 ;
   version = 4 ;
   strcpy_s ( filename , "E:\\MarketDataAssorted\\INDEXES\\$OEX.TXT" ) ;
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


/*
   The market data is read.  Initialize for gap analysis
*/

   ngaps = NGAPS ;
   k = 1 ;
   for (i=0 ; i<ngaps-1 ; i++) {
      gap_size[i] = k ;
      k *= 2 ;
      }

/*
   Compute trend and find min, max, quantile.
*/

   nind = nprices - full_lookback + 1 ;   // This many indicators

   trend = (double *) malloc ( 2 * nind * sizeof(double) ) ;
   if (trend == NULL) {
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      goto FINISH ;
      }
   trend_sorted = trend + nind ;

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
      trend_sorted[i] = trend[i] ;
      if (trend[i] < trend_min)
         trend_min = trend[i] ;
      if (trend[i] > trend_max)
         trend_max = trend[i] ;
      }

   qsortd ( 0 , nind-1 , trend_sorted ) ;
   k = (int) (fractile * (nind+1)) - 1 ;
   if (k < 0)
      k = 0 ;
   trend_quantile = trend_sorted[k] ;

   printf ( "\n\nTrend  min=%.4lf  max=%.4lf  %.3lf quantile=%.4lf",
            trend_min, trend_max, fractile, trend_quantile ) ;

/*
   Do gap analysis for trend
*/

   printf ( "\n\nGap analysis for trend with lookback=%d", lookback ) ;
   printf ( "\n  Size   Count" ) ;

   gap_analyze ( nind , trend , trend_quantile , ngaps , gap_size , gap_count ) ;

   for (i=0 ; i<ngaps ; i++) {
      if (i < ngaps-1)
         printf ( "\n %5d %7d", gap_size[i], gap_count[i] ) ;
      else
         printf ( "\n>%5d %7d", gap_size[ngaps-2], gap_count[i] ) ;
      }

/*
   Compute average true range (volatility) and find min, max, median.
*/

   volatility = (double *) malloc ( 2 * nind * sizeof(double) ) ;
   if (volatility == NULL) {
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      goto FINISH ;
      }
   volatility_sorted = volatility + nind ;

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
      volatility_sorted[i] = volatility[i] ;
      if (volatility[i] < volatility_min)
         volatility_min = volatility[i] ;
      if (volatility[i] > volatility_max)
         volatility_max = volatility[i] ;
      }

   qsortd ( 0 , nind-1 , volatility_sorted ) ;
   k = (int) (fractile * (nind+1)) - 1 ;
   if (k < 0)
      k = 0 ;
   volatility_quantile = volatility_sorted[k] ;

   printf ( "\n\nVolatility  min=%.4lf  max=%.4lf  %.3lf quantile=%.4lf",
            volatility_min, volatility_max, fractile, volatility_quantile ) ;

/*
   Do gap analysis for volatility
*/

   printf ( "\n\nGap analysis for volatility with lookback=%d", lookback ) ;
   printf ( "\n  Size   Count" ) ;

   gap_analyze ( nind , volatility , volatility_quantile , ngaps , gap_size , gap_count ) ;

   for (i=0 ; i<ngaps ; i++) {
      if (i < ngaps-1)
         printf ( "\n %5d %7d", gap_size[i], gap_count[i] ) ;
      else
         printf ( "\n>%5d %7d", gap_size[ngaps-2], gap_count[i] ) ;
      }

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

   printf ( "\n\nPress any key..." ) ;
   _getch () ;  // Wait for user to press a key

   exit ( 0 ) ;
}
