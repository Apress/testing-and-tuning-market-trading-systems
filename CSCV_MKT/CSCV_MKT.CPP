/******************************************************************************/
/*                                                                            */
/*  CSCV_MKT - Computationally symmetric cross validation.                    */
/*         This is the market-processing version.                             */
/*                                                                            */
/*  The core routine is in CSCV_CORE.CPP which handles any app                */
/*  This is the wrapper for applying CSCV to moving-average crossover         */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>

#define MKTBUF 128   /* Alloc for market info in chunks of this many records */

double criter ( int n , double *returns ) ;

double cscvcore (
   int ncases ,         // Number of columns in returns matrix (change fastest)
   int n_systems ,      // Number of rows (competitors); should be large enough to reduce granularity
   int n_blocks ,       // Number of blocks (even!) into which the cases will be partitioned
   double *returns ,    // N_systems by ncases matrix of returns, case changing fastest
   int *indices ,       // Work vector n_blocks long
   int *lengths ,       // Work vector n_blocks long
   int *flags ,         // Work vector n_blocks long
   double *work ,       // Work vector ncases long
   double *is_crits ,   // Work vector n_systems long
   double *oos_crits    // Work vector n_systems long
   ) ;


/*
--------------------------------------------------------------------------------

   Local routine computes one-bar returns for all short-term and
   long-term lookbacks of a primitive moving-average crossover system.
   The computed returns matrix has max_lookback * (max_lookback-1) / 2 rows
   and nprices-max_lookback columns, which change fastest.
   Note that this is the transpose of the matrix in the original paper.

--------------------------------------------------------------------------------
*/

void get_returns (
   int nprices ,      // Number of log prices in 'prices'
   double *prices ,   // Log prices
   int max_lookback , // Maximum lookback to use
   double *returns    // Computed matrix of returns
   )
{
   int i, j, ishort, ilong, iret ;
   double ret, long_mean, long_sum, short_mean, short_sum ;

   iret = 0 ;   // Will index computed returns

   for (ilong=2 ; ilong<=max_lookback ; ilong++) {  // Long-term lookback
      for (ishort=1 ; ishort<ilong ; ishort++) {    // Short-term lookback

         // We have a pair of lookbacks.  Compute short-term and long-term moving averages.
         // The index of the first legal bar in prices is max_lookback-1, because
         // we will need max_lookback cases (including the decision bar) 
         // in the longest long-term moving average.
         // We must stop one bar before the end of the price array because we need
         // the next price to compute the return from the decision.

         for (i=max_lookback-1 ; i<nprices-1 ; i++) { // Compute performance across history

            if (i == max_lookback-1) { // Find the short-term and long-term moving averages for the first valid case.
               short_sum = 0.0 ;                 // Cumulates short-term lookback sum
               for (j=i ; j>i-ishort ; j--)
                  short_sum += prices[j] ;
               long_sum = short_sum ;            // Cumulates long-term lookback sum
               while (j>i-ilong)
                  long_sum += prices[j--] ;
               }

            else {                               // Update the moving averages
               short_sum += prices[i] - prices[i-ishort] ;
               long_sum += prices[i] - prices[i-ilong] ;
               }

            short_mean = short_sum / ishort ;
            long_mean = long_sum / ilong ;

            // We now have the short-term and long-term moving averages ending at bar i
            // Save the return produced by this decision bar.

            if (short_mean > long_mean)       // Long position
               ret = prices[i+1] - prices[i] ;
            else if (short_mean < long_mean)  // Short position
               ret = prices[i] - prices[i+1] ;
            else
               ret = 0.0 ;

            returns[iret++] = ret ;
            } // For i (decision bar)

         } // For ishort, all short-term lookbacks
      } // For ilong, all long-term lookbacks

   assert ( iret == (max_lookback * (max_lookback-1) / 2 * (nprices - max_lookback)) ) ;
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
   int i, nprices, n_blocks, max_lookback, n_systems, n_returns ;
   int *indices, *lengths, *flags, bufcnt ;
   double *prices, *returns, *work, *is_crits, *oos_crits, prob, crit, best_crit ;
   char line[256], filename[4096], *cptr ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 4) {
      printf ( "\nUsage: CSCV_MKT  n_blocks  max_lookback  filename" ) ;
      printf ( "\n  n_blocks - number of blocks into which cases are partitioned" ) ;
      printf ( "\n  max_lookback - Maximum moving-average lookback" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   n_blocks = atoi ( argv[1] ) ;
   max_lookback = atoi ( argv[2] ) ;
   strcpy_s ( filename , argv[3] ) ;
#else
   n_blocks = 4 ;
   max_lookback = 10 ;
   strcpy_s ( filename , "E:\\MarketDataAssorted\\SP100\\$SPX.TXT" ) ;
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
      printf ( "\n\nInsufficient memory reading market history file %s", filename ) ;
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
            printf ( "\n\nInsufficient memory reading market history file %s", filename ) ;
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
   Initialize
*/

   n_returns = nprices - max_lookback ;
   n_systems = max_lookback * (max_lookback-1) / 2 ;

   if (nprices < 2  ||  n_blocks < 2  ||  max_lookback < 2  ||  n_returns < n_blocks) {
      printf ( "\nUsage: CSCV_MKT  n_blocks  max_lookback  filename" ) ;
      printf ( "\n  n_blocks - number of blocks into which cases are partitioned" ) ;
      printf ( "\n  max_lookback - Maximum moving-average lookback" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   printf ( "\n\nnprices=%d  n_blocks=%d  max_lookback=%d  n_systems=%d  n_returns=%d",
            nprices, n_blocks,  max_lookback, n_systems, n_returns ) ;

   returns = (double *) malloc ( n_systems * n_returns * sizeof(double) ) ;
   indices = (int *) malloc ( n_blocks * sizeof(int) ) ;
   lengths = (int *) malloc ( n_blocks * sizeof(int) ) ;
   flags = (int *) malloc ( n_blocks * sizeof(int) ) ;
   work = (double *) malloc ( n_returns * sizeof(double) ) ;
   is_crits = (double *) malloc ( n_systems * sizeof(double) ) ;
   oos_crits = (double *) malloc ( n_systems * sizeof(double) ) ;

/*
   Do it and finish up
*/

   get_returns ( nprices , prices , max_lookback , returns ) ;

   prob = cscvcore ( n_returns , n_systems , n_blocks , returns , indices ,
                     lengths , flags , work , is_crits , oos_crits ) ;

   // Find return of grand best system

   for (i=0 ; i<n_systems ; i++) {
      crit = criter ( n_returns , returns + i * n_returns ) ;
      if (i == 0  ||  crit > best_crit)
         best_crit = crit ;
      }

   // Done.  Print results and clean up.

   printf ( "\n\nnprices=%d  n_blocks=%d  max_lookback=%d  n_systems=%d  n_returns=%d",
            nprices, n_blocks,  max_lookback, n_systems, n_returns ) ;
   printf ( "\n1000 * Grand criterion = %.4lf  Prob = %.4lf", 1000.0 * best_crit, prob ) ;
   _getch () ;  // Wait for user to press a key

   free ( prices ) ;
   free ( returns ) ;
   free ( indices ) ;
   free ( lengths ) ;
   free ( flags ) ;
   free ( work ) ;
   free ( is_crits ) ;
   free ( oos_crits ) ;

   exit ( 0 ) ;
}
