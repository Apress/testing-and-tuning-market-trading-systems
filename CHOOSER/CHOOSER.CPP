/*****************************************************************************/
/*                                                                           */
/*  CHOOSER - This nested walkforward demonstration moves bar-to-bar.        */
/*     At each bar it examines the recent long performance of multiple       */
/*     markets, the performance that would have been obtained by simply      */
/*     buying and holding each market during the window period.              */
/*     It then purchases and holds for the next bar whichever market had     */
/*     the best recent performance.  But how do we measure performance?      */
/*     That's the selection aspect of this demonstration.  We use several    */
/*     different performance measures, and when we buy the best market for   */
/*     the next bar, we base that decision on whichever performance          */
/*     measure has the best recent track record.                             */
/*                                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include <assert.h>

#define MAX_MARKETS 1024   /* Maximum number of markets */
#define MAX_NAME_LENGTH 16 /* One more than max number of characters in a market name */
#define BLOCK_SIZE 4096    /* Number of market records allocated in chunks; not critical */
#define MAX_CRITERIA 16    /* Maximum number of criteria (each programmed separately) */


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

   Criterion function: total return (assumes 'prices' are actually log prices)

--------------------------------------------------------------------------------
*/

double total_return ( int n , double *prices )
{
   return prices[n-1] - prices[0] ;
}


/*
--------------------------------------------------------------------------------

   Criterion function: raw Sharpe ratio (assumes 'prices' are actually log prices)

--------------------------------------------------------------------------------
*/

double sharpe_ratio ( int n , double *prices )
{
   int i ;
   double diff, mean, var ;

   mean = (prices[n-1] - prices[0]) / (n - 1.0) ;

   var = 1.e-60 ;  // Ensure no division by 0 later
   for (i=1 ; i<n ; i++) {
      diff = (prices[i] - prices[i-1]) - mean ;
      var += diff * diff ;
      }

   return mean / sqrt ( var / (n-1) ) ;
}


/*
--------------------------------------------------------------------------------

   Criterion function: profit factor (assumes 'prices' are actually log prices)

--------------------------------------------------------------------------------
*/

double profit_factor ( int n , double *prices )
{
   int i ;
   double ret, win_sum, lose_sum ;

   win_sum = lose_sum = 1.e-60 ;

   for (i=1 ; i<n ; i++) {
      ret = prices[i] - prices[i-1] ;
      if (ret > 0.0)
         win_sum += ret ;
      else
         lose_sum -= ret ;
      }

   return win_sum / lose_sum ;
}


/*
--------------------------------------------------------------------------------

   Master criterion function

--------------------------------------------------------------------------------
*/

double criterion ( int which , int n , double *prices )
{
   if (which == 0)
      return total_return ( n , prices ) ;

   if (which == 1)
      return sharpe_ratio ( n , prices ) ;

   if (which == 2)
      return profit_factor ( n , prices ) ;

   return -1.e60 ;
}


/*
--------------------------------------------------------------------------------

   Permutation routines (assumes prices in 'data' are actually log prices)

   We have nc cases from index 0 through nc-1.
   Case 'offset' is the first case that will change, and offset must be positive
   because the case at offset-1 is the 'basis' case and remains unchanged.
   The last case examined is at nc-1, but it, too, will remain unchanged.
   Thus, the shuffled array starts and ends at their original values.
   Only the interior elements change.

   If a dataset is permuted in separate sections, the sections must not overlap.
   The 'basis' case at offset-1 is included in the region that must not overlap.

   For example, we could permute with offset=1 (must be greater than 0) and
   nc=5.  Cases 1 through 3 would then change.  (The first (0) and last (4) cases
   remain the same.)
   A subsequent permute must then begin at offset=5 or more.
   This is because Case 4 is not changed by either permute operation.

--------------------------------------------------------------------------------
*/

void prepare_permute (
   int nc ,           // Number of cases total (not just starting at offset)
   int nmkt ,         // Number of markets
   int offset ,       // Index of first case to be permuted (>0)
   double **data ,    // Input of nmkt by nc price matrix
   double **changes   // Work area; returns computed changes
   )
{
   int icase, imarket ;

   for (imarket=0 ; imarket<nmkt ; imarket++) {
      for (icase=offset ; icase<nc ; icase++)
         changes[imarket][icase] = data[imarket][icase] - data[imarket][icase-1] ;
      }
}

void do_permute (
   int nc ,           // Number of cases total (not just starting at offset)
   int nmkt ,         // Number of markets
   int offset ,       // Index of first case to be permuted (>0)
   double **data ,    // Returns nmkt by nc shuffled price matrix
   double **changes   // Work area; computed changes from prepare_permute
   )
{
   int i, j, icase, imarket ;
   double dtemp ;

   // Shuffle the changes, permuting each market the same to preserve correlations
   // We do not include the first case in the shuffling, as it is the starting price, not a change

   i = nc-offset ;        // Number remaining to be shuffled
   while (i > 1) {        // While at least 2 left to shuffle
      j = (int) (unifrand() * i) ;
      if (j >= i)         // Should never happen, but be safe
         j = i - 1 ;
      --i ;
      for (imarket=0 ; imarket<nmkt ; imarket++) {
         dtemp = changes[imarket][i+offset] ;
         changes[imarket][i+offset] = changes[imarket][j+offset] ;
         changes[imarket][j+offset] = dtemp ;
         }
      } // Shuffle the changes

   // Now rebuild the prices, using the shuffled changes

   for (imarket=0 ; imarket<nmkt ; imarket++) {
      for (icase=offset ; icase<nc ; icase++)
         data[imarket][icase] = data[imarket][icase-1] + changes[imarket][icase] ;
      }
}


/*
--------------------------------------------------------------------------------

   Main program

--------------------------------------------------------------------------------
*/

void main ( int argc , char *argv[] )
{
   int i, j, k, return_value, n_markets, full_date, year, month, day ;
   int line_number, date, max_date, all_same_date, n_cases ;
   int itemp, prior_date, **market_date, *market_index, *market_n, grand_index, n_allocated ;
   int IS_n, OOS1_n, IS_start, OOS1_start, OOS1_end, OOS2_start, OOS2_end ;
   int icrit, imarket, n_criteria, ibest, ibestcrit, irep, nreps, crit_pval[MAX_CRITERIA], final_pval ;
   int crit_count[MAX_CRITERIA] ;
   double open, high, low, close, **market_close, crit, best_crit, sum, ret, crit_perf[MAX_CRITERIA], final_perf ;
   double *OOS1, *OOS2, **permute_work, perf ;
   char FileListName[1024], MarketFileName[1024], line[512], msg[256], *lptr ;
   char *market_names ;
   FILE *fpReport, *fpList, *fpMarket ;

   return_value = 0 ;

#if 1
   if (argc != 5) {
      printf ( "\nUSAGE: CHOOSER FileList IS_n OOS1_n nreps" ) ;
      printf ( "\n  FileList - Text file containing list of competing market history files" ) ;
      printf ( "\n  IS_n - N of market history records for each selection criterion to analyze" ) ;
      printf ( "\n  OOS1_n - N of OOS records for choosing best criterion" ) ;
      printf ( "\n  nreps - Number of Monte-Carlo replications (1 or 0 for none)" ) ;
      exit ( 0 ) ;
      }
   strcpy_s ( FileListName , argv[1] ) ;
   IS_n = atoi ( argv[2] ) ;
   OOS1_n = atoi ( argv[3] ) ;
   nreps = atoi ( argv[4] ) ;
#else
   strcpy_s ( FileListName , "d:\\validate\\chooser1\\BigStuff.txt" ) ;
   IS_n = 200 ;
   IS_n = 1000 ;
   OOS1_n = 100 ;
   nreps = 3 ;
#endif

   if (nreps < 1)
      nreps = 1 ;

   if (IS_n < 2  ||  OOS1_n < 1) {
      printf ( "\nUSAGE: CHOOSER FileList IS_n OOS1_n nreps" ) ;
      printf ( "\n  FileList - Text file containing list of competing market history files" ) ;
      printf ( "\n  IS_n - N of market history records for each selection criterion to analyze" ) ;
      printf ( "\n  OOS1_n - N of OOS records for choosing best criterion" ) ;
      printf ( "\n  nreps - Number of Monte-Carlo replications (1 or 0 for none)" ) ;
      exit ( 0 ) ;
      }


/*
--------------------------------------------------------------------------------

   Open the report and market list files, then allocate memory.
   Initialize all memory pointers to NULL to prevent attempts to free them at
   the end if we have an error.  We really should do better memory checking
   instead of just asserts.

--------------------------------------------------------------------------------
*/

   market_names = NULL ;
   market_date = NULL ;
   market_index = NULL ;
   market_n = NULL ;
   market_close = NULL ;
   permute_work = NULL ;
   OOS1 = NULL ;
   OOS2 = NULL ;

   if (fopen_s ( &fpReport , "CHOOSER.LOG" , "wt" )) {
      printf ( "\nERROR... Cannot open REPORT.LOG for writing" ) ;
      exit ( 1 ) ;
      }

   if (fopen_s ( &fpList , FileListName , "rt" )) {
      printf ( "\nERROR... Cannot open list file %s", FileListName ) ;
      fpList = NULL ;
      return_value = 1 ;
      goto FINISH ;
      }

   market_names = (char *) malloc ( MAX_MARKETS * MAX_NAME_LENGTH * sizeof(char) ) ;
   assert ( market_names != NULL ) ;

   market_date = (int **) malloc ( MAX_MARKETS * sizeof(int *) ) ;
   assert ( market_date != NULL ) ;

   market_index = (int *) malloc ( MAX_MARKETS * sizeof(int) ) ;
   assert ( market_index != NULL ) ;

   market_n = (int *) malloc ( MAX_MARKETS * sizeof(int) ) ;
   assert ( market_n != NULL ) ;

   market_close = (double **) malloc ( MAX_MARKETS * sizeof(double *) ) ;
   assert ( market_close != NULL ) ;

   for (i=0 ; i<MAX_MARKETS ; i++) {
      market_date[i] = NULL ;
      market_close[i] = NULL ;
      }

   fprintf ( fpReport, "CHOOSER log with IS_n=%d  OOS1_n=%d  Reps=%d", IS_n, OOS1_n, nreps ) ;


/*
--------------------------------------------------------------------------------

   Read the market data, one market at a time

--------------------------------------------------------------------------------
*/

   n_markets = 0 ;
   for (;;) {

      // Get the name of a market file
      if ((fgets ( line , 256 , fpList ) == NULL) || (strlen ( line ) < 2)) {
         if (ferror ( fpList )  ||  ! n_markets) {
            printf ( "\nERROR... Cannot read list file %s", FileListName ) ;
            return_value = 1 ;
            goto FINISH ;
            }
         else
            break ;       // Normal end of list file
         }

      // Copy this market file name

      lptr = &line[0] ;
      k = 0 ;
      while (isalnum(*lptr)  ||  *lptr == '_'  ||  *lptr == '\\'  ||  *lptr == ':'  ||  *lptr == '.')
         MarketFileName[k++] = *lptr++ ;
      MarketFileName[k] = 0 ;

      // Get and save the name of the market from the file name
      // We assume it is just before the last period.

      strcpy_s ( msg , MarketFileName ) ;
      lptr = &msg[k-1] ;  // Last character in file name
      while (lptr > &msg[0]  &&  *lptr != '.')
         --lptr ;
      if (*lptr != '.') {
         printf ( "\nERROR... Market file name (%s) is not legal", MarketFileName ) ;
         return_value = 1 ;
         goto FINISH ;
         }
      *lptr = 0 ;   // This removes extension
      while (lptr > &msg[0]  &&  *lptr != '.'  &&  *lptr != '\\'  &&  *lptr != ':')
         --lptr ;   // Back up until we get path stuff
      if (*lptr == '.'  ||  *lptr == '\\'  ||  *lptr == ':')  // If a path character caused loop exit, pass it
         ++lptr ;
      if (strlen ( lptr ) > MAX_NAME_LENGTH-1) {
         printf ( "\nERROR... Market name (%s) is too long", lptr ) ;
         return_value = 1 ;
         goto FINISH ;
         }
      strcpy_s ( market_names+n_markets*MAX_NAME_LENGTH , MAX_NAME_LENGTH , lptr ) ;

/*
   We now have the name of a market history file.  Read this file.
*/

      if (fopen_s ( &fpMarket , MarketFileName , "rt" )) {
         printf ( "\nERROR... Cannot open market file %s", MarketFileName ) ;
         return_value = 1 ;
         goto FINISH ;
         }

      printf ( "\nReading market file %s...", MarketFileName ) ;

      line_number = prior_date = n_allocated = 0 ;
      for (;;) {

         if ((fgets ( line , 512 , fpMarket ) == NULL) || (strlen ( line ) < 2)) {
            if (ferror ( fpMarket )  ||  ! line_number) {
               fclose ( fpMarket ) ;
               printf ( "\nERROR... Cannot read market file %s", MarketFileName ) ;
               return_value = 1 ;
               goto FINISH ;
               }
            else
               break ;       // Normal end of file
            }
         
         // Get the date, open, high, low, close.

         // Pass any leading delimeters, and then copy the field up to the next delimeter
         // Date

         lptr = line ;
         while (*lptr == ' ' || *lptr == ',' || *lptr == '\t') // Skip delimiters
            ++lptr ;
         k = 0 ;
         while (*lptr  &&  *lptr != ' '  &&  *lptr != ','  &&  *lptr != '\t')
            msg[k++] = *lptr++ ;
         msg[k] = 0 ;

         full_date = itemp = atoi ( msg ) ;
         year = itemp / 10000 ;
         itemp -= year * 10000 ;
         month = itemp / 100 ;
         itemp -= month * 100 ;
         day = itemp ;

         if (month < 1 || month > 12 || day < 1 || day > 31 || year < 1800 || year > 2030) {
            fclose ( fpMarket ) ;
            printf ( "\nERROR... Invalid date %d in market file %s line %d",
                     full_date, MarketFileName, line_number+1 ) ;
            return_value = 1 ;
            goto FINISH ;
            }

         if (full_date <= prior_date) {
            fclose ( fpMarket ) ;
            printf ( "\nERROR... Date failed to increase in market file %s line %d",
                     MarketFileName, line_number+1 ) ;
            return_value = 1 ;
            goto FINISH ;
            }

         prior_date = full_date ;



         // Open
         while (*lptr == ' ' || *lptr == ',' || *lptr == '/' || *lptr == '\t') // Skip delimiters
            ++lptr ;
         k = 0 ;
         while (*lptr  &&  *lptr != ' '  &&  *lptr != ','  &&  *lptr != '/'  &&  *lptr != '\t')
            msg[k++] = *lptr++ ;
         msg[k] = 0 ;
         open = (float) atof ( msg ) ;

         // High
         while (*lptr == ' ' || *lptr == ',' || *lptr == '/' || *lptr == '\t') // Skip delimiters
            ++lptr ;
         if (strlen(lptr)  &&  isprint(*lptr)) {
            k = 0 ;
            while (*lptr  &&  *lptr != ' '  &&  *lptr != ','  &&  *lptr != '/'  &&  *lptr != '\t')
               msg[k++] = *lptr++ ;
            msg[k] = 0 ;
            high = (float) atof ( msg ) ;
            }
         else
            high = open ;

         // Low
         while (*lptr == ' ' || *lptr == ',' || *lptr == '/' || *lptr == '\t') // Skip delimiters
            ++lptr ;
         if (strlen(lptr)  &&  isprint(*lptr)) {
            k = 0 ;
            while (*lptr  &&  *lptr != ' '  &&  *lptr != ','  &&  *lptr != '/'  &&  *lptr != '\t')
               msg[k++] = *lptr++ ;
            msg[k] = 0 ;
            low = (float) atof ( msg ) ;
            }
         else
            low = open ;

         // Close
         while (*lptr == ' ' || *lptr == ',' || *lptr == '/' || *lptr == '\t') // Skip delimiters
            ++lptr ;
         if (strlen(lptr)  &&  isprint(*lptr)) {
            k = 0 ;
            while (*lptr  &&  *lptr != ' '  &&  *lptr != ','  &&  *lptr != '/'  &&  *lptr != '\t')
               msg[k++] = *lptr++ ;
            msg[k] = 0 ;
            close = (float) atof ( msg ) ;
            }
         else
            close = open ;

         if (high < open  ||  high < close  ||  low > open  ||  low > close) {
            fclose ( fpMarket ) ;
            printf ( "\nERROR... Open or close outside high/low bounds in market file %s line %d",
                     MarketFileName, line_number+1 ) ;
            return_value = 1 ;
            goto FINISH ;
            }

         // Allocate another block of memory if needed

         if (line_number >= n_allocated) {
            market_date[n_markets] = (int *) realloc ( market_date[n_markets] , (n_allocated+BLOCK_SIZE) * sizeof(int) ) ;
            assert ( market_date[n_markets] != NULL ) ;
            market_close[n_markets] = (double *) realloc ( market_close[n_markets] , (n_allocated+BLOCK_SIZE) * sizeof(double) ) ;
            assert ( market_close[n_markets] != NULL ) ;
            n_allocated += BLOCK_SIZE ;
            }

         // Save this market data
         market_date[n_markets][line_number] = full_date ;
         market_close[n_markets][line_number] = close ;

         ++line_number ;  // Also counts market records
         } // For all lines in the market history file

      fclose ( fpMarket ) ;

      market_n[n_markets] = line_number ;

      fprintf ( fpReport, "\nMarket file %s had %d records from date %d to %d",
      MarketFileName, line_number, market_date[n_markets][0], market_date[n_markets][line_number-1] ) ;

      ++n_markets ;
      } // For all lines in the market list file

   fclose ( fpList ) ;
   fpList = NULL ;


/*
-----------------------------------------------------------------------------------------

   We have completely finished reading all markets, and their data is stored.
   However, our upcoming studies require that all data be date aligned, and
   there is no guaranty that we have this.
   So now we remove all records that do not have data for all markets.
   The algorithm given here may not be the most efficient way to do this,
   but compared with what's coming soon this is a drop in the bucket of CPU time.

   The algorithm is as follows:
      Initialize each market's current index to 0
      Initialize the grand (compressed) index to 0
      Loop
         Find the latest (largest) date at each market's current index across all markets
         Advance all markets' current index until the date reaches or passes this date
         If all markets have the same current date:
            keep this date by copying this market record to the grand index spot
            Advance each market's current index as well as the grand index

-----------------------------------------------------------------------------------------
*/

   for (i=0 ; i<n_markets ; i++)
      market_index[i] = 0 ;
   grand_index = 0 ;

   for (;;) {

      // Find max date at current index of each market

      max_date = 0 ;
      for (i=0 ; i<n_markets ; i++) {
         date = market_date[i][market_index[i]] ;
         if (date > max_date)
            max_date = date ;
         }

      // Advance all markets until they reach or pass max date
      // Keep track of whether they all equal max_date

      all_same_date = 1 ;
      for (i=0 ; i<n_markets ; i++) {
         while (market_index[i] < market_n[i]) {    // Must not over-run a market!
            date = market_date[i][market_index[i]] ;
            if (date >= max_date)
               break ;
            ++market_index[i] ;
            }
         if (date != max_date)                // Did some market jump over max?
            all_same_date = 0 ;

         if (market_index[i] >= market_n[i])  // If even one market runs out
            break ;                           // We are done
         }

      if (i < n_markets)  // If even one market runs out
         break ;          // We are done

      // If we have a complete set for this date, grab it

      if (all_same_date) {
         for (i=0 ; i<n_markets ; i++) {
            assert ( market_date[i][market_index[i]] = max_date ) ;
            market_date[i][grand_index] = max_date ;  // Redundant, but clear
            market_close[i][grand_index] = market_close[i][market_index[i]] ;
            ++market_index[i] ;
            }
         ++grand_index ;
         }
      }

   n_cases = grand_index ;

   fprintf ( fpReport, "\n\nMerged database has %d records from date %d to %d",
             n_cases, market_date[0][0], market_date[0][n_cases-1] ) ;

/*
   Free memory that we no longer need
*/

   for (i=0 ; i<n_markets ; i++) {  // We don't need the dates any more
      if (market_date[i] != NULL)
         free ( market_date[i] ) ;
      }

   free ( market_date ) ;
   market_date = NULL ;
   free ( market_n ) ;
   market_n = NULL ;
   free ( market_index ) ;
   market_index = NULL ;

/*
   Allocate memory for permutation if requested
*/

   if (nreps > 1) {
      permute_work = (double **) malloc ( n_markets * sizeof(double *) ) ;
      assert ( permute_work != NULL ) ;

      for (i=0 ; i<n_markets ; i++) {
         permute_work[i] = (double *) malloc ( n_cases * sizeof(double) ) ;
         assert ( permute_work[i] != NULL ) ;
         }
      }


/*
-----------------------------------------------------------------------------------------

   We now have a matrix of closing prices.
   It has n_cases rows and n_markets columns.  It's time for the interesting stuff.

   IS_n - N of market history records for each selection criterion to analyze
   OOS1_n - N of OOS selection criteria for choosing best criterion
   IS_start - Starting index of current market performance window
   OOS1_start - Starting index of current OOS set 1 (advances with window)
   OOS1_end - And one past its last case (also serves as current OOS1 case index)
   OOS2_start - Starting index of complete OOS set 2; fixed at IS_n + OOS1_n
   OOS2_end - And one past its last case (also serves as current OOS2 case index)

-----------------------------------------------------------------------------------------
*/

/*
   Convert all closing prices to log prices.
   This saves enormous CPU time by avoiding having to take logs
   when we evaluate the criteria (which happens many, many times!).
*/

   for (imarket=0 ; imarket<n_markets ; imarket++) {
      for (i=0 ; i<n_cases ; i++)
         market_close[imarket][i] = log ( market_close[imarket][i] ) ;
      }

/*
   Print return of each market over the OOS2 period.
   Note that the first OOS2 return will be relative to case OOS2_start-1,
   and OOS2_start = IS_n + OOS1_n.
   We multiply the printed value by 25200 to make it roughly annualized percent return.
*/

   fprintf ( fpReport, "\n\n25200 * mean return of each market in OOS2 period..." ) ;
   sum = 0.0 ;
   for (i=0 ; i<n_markets ; i++) {
      ret = 25200 * (market_close[i][n_cases-1] - market_close[i][IS_n+OOS1_n-1]) / (n_cases - IS_n - OOS1_n) ;
      sum += ret ;
      fprintf ( fpReport, "\n%15s %9.4lf", &market_names[i*MAX_NAME_LENGTH], ret ) ;
      }
   fprintf ( fpReport, "\nMean = %9.4lf", sum / n_markets ) ;

/*
   Allocate memory for OOS1 and OOS2
*/

   n_criteria = 3 ;

   OOS1 = (double *) malloc ( n_criteria * n_cases * sizeof(double) ) ;
   assert ( OOS1 != NULL ) ;

   OOS2 = (double *) malloc ( n_cases * sizeof(double) ) ;
   assert ( OOS2 != NULL ) ;


/*
-----------------------------------------------------------------------------------------

   The Monte-Carlo permutation loop begins here, after some small initializations

-----------------------------------------------------------------------------------------
*/

   printf ( "\n\nComputing" ) ;

   for (i=0 ; i<n_criteria ; i++)
      crit_count[i] = 0 ;   // Counts how many times each criterion is chosen

   if (nreps > 1) {
      prepare_permute ( IS_n , n_markets , 1 , market_close , permute_work ) ;
      prepare_permute ( IS_n+OOS1_n , n_markets , IS_n , market_close , permute_work ) ;
      prepare_permute ( n_cases , n_markets , IS_n+OOS1_n , market_close , permute_work ) ;
      }


   for (irep=0 ; irep<nreps ; irep++) {

      if (irep) {  // Permute after the first replication
         do_permute ( IS_n , n_markets , 1 , market_close , permute_work ) ;
         do_permute ( IS_n+OOS1_n , n_markets , IS_n , market_close , permute_work ) ;
         do_permute ( n_cases , n_markets , IS_n+OOS1_n , market_close , permute_work ) ;
         }


/*
   Initialize indices
*/

      IS_start = 0 ;                 // Start market window with first case
      OOS1_start = OOS1_end = IS_n ; // First OOS1 case is right after first price set
      OOS2_start = OOS2_end = IS_n + OOS1_n ;  // First OOS2 case is right after first complete OOS1

      printf ( "." ) ;

/*
   Main outermost loop traverses market history
*/

      for (;;) {

         // Evaluate all performance criteria for all markets
         // For each criterion find the best market and
         // save the return of the next record for each criterion

         for (icrit=0 ; icrit<n_criteria ; icrit++) {
            best_crit = -1.e60 ;
            for (imarket=0 ; imarket<n_markets ; imarket++) {
               crit = criterion ( icrit , IS_n , market_close[imarket]+IS_start ) ;
               if (crit > best_crit) {
                  best_crit = crit ;
                  ibest = imarket ;
                  }
               }
            OOS1[icrit*n_cases+OOS1_end] = market_close[ibest][OOS1_end] - market_close[ibest][OOS1_end-1] ;
            }

         if (OOS1_end >= n_cases-1)  // Have we hit the end of the data?
            break ;                  // Stop due to lack of another for OOS2

         // Advance the window: first half of advance
         // If we do not yet have enough OOS cases in OOS1, keep filling it.

         ++IS_start ;  // Advance training window
         ++OOS1_end ;  // Advance current OOS1 case

         if (OOS1_end - OOS1_start < OOS1_n)  // Are we still filling OOS1?
            continue ;  // We cannot proceed until we have enough cases to compute an OOS2 return

         // When we get here we have enough cases in OOS1 to compute an OOS2 case.
         // Find the best criterion in OOS1 and use this criterion to find the best market.
         // We just use total OOS1 total return as our secondary selection criterion.
         // Feel free to change this if you wish.
         // Then compute the return of the best market according to this criterion.
         // Remember that OOS1_end now points one past what we have in OOS1.


         best_crit = -1.e60 ;
         for (icrit=0 ; icrit<n_criteria ; icrit++) {  // Find the best criterion using OOS1
            crit = 0.0 ;
            for (i=OOS1_start ; i<OOS1_end ; i++)
               crit += OOS1[icrit*n_cases+i] ;
            if (crit > best_crit) {
               best_crit = crit ;
               ibestcrit = icrit ;
               }
            }

         if (irep == 0)
            ++crit_count[ibestcrit] ;   // This is purely for user's edification

         best_crit = -1.e60 ;
         for (imarket=0 ; imarket<n_markets ; imarket++) { // Use recently best criterion to select market
            crit = criterion ( ibestcrit , IS_n , market_close[imarket]+OOS2_end-IS_n ) ;
            if (crit > best_crit) {
               best_crit = crit ;
               ibest = imarket ;
               }
            }

         // This is a strictly long test, long some market every bar
         OOS2[OOS2_end] = market_close[ibest][OOS2_end] - market_close[ibest][OOS2_end-1] ;
         ++OOS1_start ;   // Finish advancing window across market history
         ++OOS2_end ;
         } // Main loop that traverses market history

      assert ( OOS1_end == n_cases - 1 ) ; // We exited loop before advancing this
      assert ( OOS2_end == n_cases ) ;

/*
   Compute and save mean market performance of each criterion
   We examine the same bars for which we have OOS2 data, making them commensurate.
   Perform solo MCPT tests.
*/

      for (i=0 ; i<n_criteria ; i++) {
         sum = 0.0 ;
         for (j=OOS2_start ; j<OOS2_end ; j++)
            sum += OOS1[i*n_cases+j] ;
         perf = 25200 * sum / (OOS2_end - OOS2_start) ;
         if (irep == 0) {
            crit_pval[i] = 1 ;
            crit_perf[i] = perf ;
            }
         else if (perf >= crit_perf[i])
            ++crit_pval[i] ;
         }

/*
   Compute and save final return; tally for grand MCPT.
   Instead of or in addition to the mean return, we might want to compute
   some other performance measure for the returns in OOS2.  Feel free.
*/

      sum = 0.0 ;
      for (i=OOS2_start ; i<OOS2_end ; i++)
         sum += OOS2[i] ;
      perf = 25200 * sum / (OOS2_end - OOS2_start) ;
      if (irep == 0) {
         final_pval = 1 ;
         final_perf = perf ;
         }
      else if (perf >= final_perf)
         ++final_pval ;

      } // For irep (Monte-Carlo replications)

/*
   Print summary information
*/

   if (nreps > 1)
      fprintf ( fpReport, "\n\n25200 * mean return of each criterion, p-value, and percent of times chosen..." ) ;
   else
      fprintf ( fpReport, "\n\n25200 * mean return of each criterion, and percent of times chosen..." ) ;

   sum = 0.0 ;
   for (i=0 ; i<n_criteria ; i++)
      sum += crit_count[i] ;

   for (i=0 ; i<n_criteria ; i++) {
      if (i == 0)
         strcpy_s ( msg , "Total return" ) ;
      else if (i == 1)
         strcpy_s ( msg , "Sharpe ratio" ) ;
      else if (i == 2)
         strcpy_s ( msg , "Profit factor" ) ;
      else
         strcpy_s ( msg , "ERROR" ) ;
      if (nreps > 1)
         fprintf ( fpReport, "\n%15s %9.4lf  p=%.3lf  Chosen %.1lf pct",
                   msg, crit_perf[i], (double) crit_pval[i] / nreps, 100.0 * crit_count[i] / sum ) ;
      else
         fprintf ( fpReport, "\n%15s %9.4lf  Chosen %.1lf pct",
                   msg, crit_perf[i], 100.0 * crit_count[i] / sum ) ;
      }

   if (nreps > 1)
      fprintf ( fpReport, "\n\n25200 * mean return of final system = %.4lf  p=%.3lf", final_perf, (double) final_pval / nreps ) ;
   else
      fprintf ( fpReport, "\n\n25200 * mean return of final system = %.4lf", final_perf ) ;

FINISH:
   if (fpList != NULL)
      fclose ( fpList ) ;

   for (i=0 ; i<n_markets ; i++) {
      if (market_date != NULL  &&  market_date[i] != NULL)
         free ( market_date[i] ) ;
      if (market_close != NULL  &&  market_close[i] != NULL)
         free ( market_close[i] ) ;
      if (permute_work != NULL  &&  permute_work[i] != NULL)
         free ( permute_work[i] ) ;
      }

   if (market_names != NULL)
      free ( market_names ) ;
   if (market_date != NULL)
      free ( market_date ) ;
   if (market_n != NULL)
      free ( market_n ) ;
   if (market_index != NULL)
      free ( market_index ) ;
   if (market_close != NULL)
      free ( market_close ) ;
   if (permute_work != NULL)
      free ( permute_work ) ;
   if (OOS1 != NULL)
      free ( OOS1 ) ;
   if (OOS2 != NULL)
      free ( OOS2 ) ;

   fclose ( fpReport ) ;

   exit ( return_value ) ;
}
