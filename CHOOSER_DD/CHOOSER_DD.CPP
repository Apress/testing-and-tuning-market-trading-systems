/*****************************************************************************/
/*                                                                           */
/*  CHOOSER_DD - This nested walkforward demonstration moves bar-to-bar.     */
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
/*  This is exactly the same trading system as that in the CHOOSER program.  */
/*  However, this does not perform a permutation test of significance,       */
/*  and this does compute and print bounds for future drawdown.              */
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

double unifrand () ;
void qsortd ( int first , int last , double *data ) ;

#define MAX_MARKETS 1024   /* Maximum number of markets */
#define MAX_NAME_LENGTH 16 /* One more than max number of characters in a market name */
#define BLOCK_SIZE 4096    /* Number of market records allocated in chunks; not critical */
#define MAX_CRITERIA 16    /* Maximum number of criteria (each programmed separately) */


/*
--------------------------------------------------------------------------------

   Compute drawdown

   This assumes that the trades are log of equity changes.
   It returns percent drawdown.

--------------------------------------------------------------------------------
*/

double drawdown (
   int n ,           // Number of trades
   double *trades    // They are here
   )
{
   int icase ;
   double cumulative, max_price, loss, dd ;

   cumulative = max_price = trades[0] ;
   dd = 0.0 ;

   for (icase=1 ; icase<n ; icase++) {
      cumulative += trades[icase] ;
      if (cumulative > max_price)
         max_price = cumulative ;
      else {
         loss = max_price - cumulative ;
         if (loss > dd)
            dd = loss ;
         }
      } // For all cases

   return 100.0 * (1.0 - exp ( -dd )) ; // Convert log change to percent
}


/*
--------------------------------------------------------------------------------

   Compute four drawdown quantiles

--------------------------------------------------------------------------------
*/

void drawdown_quantiles (
   int n_changes ,       // Number of price changes (available history)
   int n_trades ,        // Number of trades in drawdown period (<= n_changes)
   double *b_changes ,   // n_changes bootstrap sample changes supplied here
   int nboot ,           // Number of bootstraps used to compute quantiles
   double *quantsample , // Work area n_trades long
   double *work ,        // Work area nboot long
   double *q001 ,
   double *q01 ,
   double *q05 ,
   double *q10
   )
{
   int i, k, iboot ;

   for (iboot=0 ; iboot<nboot ; iboot++) {
      for (i=0 ; i<n_trades ; i++) {
         k = (int) (unifrand() * n_changes) ;
         if (k >= n_changes)
            k = n_changes - 1 ;
         quantsample[i] = b_changes[k] ;
         }
      work[iboot] = drawdown ( n_trades , quantsample ) ;
      }

   qsortd ( 0 , nboot-1 , work ) ;

   k = (int) (0.999 * (nboot+1) ) - 1 ;
   if (k < 0)
      k = 0 ;
   *q001 = work[k] ;

   k = (int) (0.99 * (nboot+1) ) - 1 ;
   if (k < 0)
      k = 0 ;
   *q01 = work[k] ;

   k = (int) (0.95 * (nboot+1) ) - 1 ;
   if (k < 0)
      k = 0 ;
   *q05 = work[k] ;

   k = (int) (0.90 * (nboot+1) ) - 1 ;
   if (k < 0)
      k = 0 ;
   *q10 = work[k] ;
}


/*
--------------------------------------------------------------------------------

   Find a quantile

--------------------------------------------------------------------------------
*/

static double find_quantile ( int n , double *data , double frac )
{
   int k ;

   k = (int) (frac * (n+1) ) - 1 ;
   if (k < 0)
      k = 0 ;
   return data[k] ;
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

   Main program

--------------------------------------------------------------------------------
*/

void main ( int argc , char *argv[] )
{
   int i, j, k, n, iboot, return_value, n_markets, full_date, year, month, day ;
   int line_number, date, max_date, all_same_date, n_cases, divisor ;
   int itemp, prior_date, **market_date, *market_index, *market_n, grand_index, n_allocated ;
   int IS_n, OOS1_n, IS_start, OOS1_start, OOS1_end, OOS2_start, OOS2_end ;
   int icrit, imarket, n_criteria, ibest, ibestcrit ;
   int crit_count[MAX_CRITERIA], bootstrap_reps, quantile_reps, n_trades ;
   double open, high, low, close, **market_close, crit, best_crit, sum, ret, crit_perf[MAX_CRITERIA], final_perf ;
   double *OOS1, *OOS2, **permute_work, perf, *bootsample, *quantile_sample, *work ;
   double *q001, *q01, *q05, *q10 ;
   char FileListName[1024], MarketFileName[1024], line[512], msg[256], *lptr ;
   char *market_names ;
   FILE *fpReport, *fpList, *fpMarket ;

   return_value = 0 ;

#if 1
   if (argc != 4) {
      printf ( "\nUSAGE: CHOOSER FileList IS_n OOS1_n" ) ;
      printf ( "\n  FileList - Text file containing list of competing market history files" ) ;
      printf ( "\n  IS_n - N of market history records for each selection criterion to analyze" ) ;
      printf ( "\n  OOS1_n - N of OOS records for choosing best criterion" ) ;
      exit ( 0 ) ;
      }
   strcpy_s ( FileListName , argv[1] ) ;
   IS_n = atoi ( argv[2] ) ;
   OOS1_n = atoi ( argv[3] ) ;
#else
   strcpy_s ( FileListName , "d:\\StatAlgs\\Drawdown\\chooser_dd\\BigStuff.txt" ) ;
   IS_n = 1000 ;
   OOS1_n = 100 ;
#endif

   if (IS_n < 2  ||  OOS1_n < 1) {
      printf ( "\nUSAGE: CHOOSER FileList IS_n OOS1_n" ) ;
      printf ( "\n  FileList - Text file containing list of competing market history files" ) ;
      printf ( "\n  IS_n - N of market history records for each selection criterion to analyze" ) ;
      printf ( "\n  OOS1_n - N of OOS records for choosing best criterion" ) ;
      exit ( 0 ) ;
      }

/*
   Aritrary user values
*/

   n_criteria = 3 ;
   bootstrap_reps = 2000 ;  // Should be at least this for good accuracy
   quantile_reps = 10000 ;  // Should be at least this for good accuracy
   n_trades = 63 ;   // One quarter if daily prices
   n_trades = 252 ;  // One year if daily prices

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

   fprintf ( fpReport, "CHOOSER_DD  log with IS_n=%d  OOS1_n=%d", IS_n, OOS1_n) ;


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

   printf ( "\n\nAligning dates..." ) ;

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
         if (date != max_date)
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
   Allocate memory for OOS1 and OOS2 and drawdown stuff
*/

   OOS1 = (double *) malloc ( n_criteria * n_cases * sizeof(double) ) ;
   assert ( OOS1 != NULL ) ;

   OOS2 = (double *) malloc ( n_cases * sizeof(double) ) ;
   assert ( OOS2 != NULL ) ;

   bootsample = (double *) malloc ( n_cases * sizeof(double) ) ;
   assert ( bootsample != NULL ) ;

   quantile_sample = (double *) malloc ( n_trades * sizeof(double) ) ;
   assert ( quantile_sample != NULL ) ;

   work = (double *) malloc ( quantile_reps * sizeof(double) ) ;
   assert ( work != NULL ) ;

   q001 = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   assert ( q001 != NULL ) ;

   q01 = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   assert ( q001 != NULL ) ;

   q05 = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   assert ( q001 != NULL ) ;

   q10 = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   assert ( q001 != NULL ) ;

/*
   Initialize
*/

   for (i=0 ; i<n_criteria ; i++)
      crit_count[i] = 0 ;   // Counts how many times each criterion is chosen

   IS_start = 0 ;                 // Start market window with first case
   OOS1_start = OOS1_end = IS_n ; // First OOS1 case is right after first price set
   OOS2_start = OOS2_end = IS_n + OOS1_n ;  // First OOS2 case is right after first complete OOS1

/*
   Main outermost loop traverses market history
*/

   printf ( "\n\nComputing trades..." ) ;

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
 */

   for (i=0 ; i<n_criteria ; i++) {
      sum = 0.0 ;
      for (j=OOS2_start ; j<OOS2_end ; j++)
         sum += OOS1[i*n_cases+j] ;
      perf = 25200 * sum / (OOS2_end - OOS2_start) ;
      crit_perf[i] = perf ;
      }

/*
   Compute and save final return
   Instead of or in addition to the mean return, we might want to compute
   some other performance measure for the returns in OOS2.  Feel free.
*/

   sum = 0.0 ;
   for (i=OOS2_start ; i<OOS2_end ; i++)
      sum += OOS2[i] ;
   perf = 25200 * sum / (OOS2_end - OOS2_start) ;
   final_perf = perf ;

/*
   Print summary information
*/

   fprintf ( fpReport, "\n\n25200 * mean log return of each criterion, and pct times chosen" ) ;

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
      fprintf ( fpReport, "\n%15s %9.4lf  Chosen %.1lf pct",
                msg, crit_perf[i], 100.0 * crit_count[i] / sum ) ;
      }

   fprintf ( fpReport, "\n\n25200 * mean return of final system = %.4lf", final_perf ) ;

/*
   Compute and print drawdown information
*/

   n = OOS2_end - OOS2_start ;
   divisor = bootstrap_reps / 10 ;   // For progress report only
   if (divisor < 1)
      divisor = 1 ;
   printf ( "\n\nDoing bootstrap" ) ;
   for (iboot=0 ; iboot<bootstrap_reps ; iboot++) {
      if (iboot % divisor == 0)
         printf ( "." ) ;
      for (i=0 ; i<n ; i++) {             // Collect a bootstrap sample from the entire OOS set
         k = (int) (unifrand() * n) ;
         if (k >= n)
            k = n - 1 ;
         bootsample[i] = OOS2[k+OOS2_start] ;
         }

      // Compute our four statistics whose bounds are being found with percentile bootstrap
      drawdown_quantiles ( n , n_trades , bootsample , quantile_reps , quantile_sample , work ,
                           &q001[iboot] , &q01[iboot] ,&q05[iboot] ,&q10[iboot] ) ;
      } // End of correct method bootstrap loop

   // Sort for CDF and find quantiles
   qsortd ( 0 , bootstrap_reps-1 , q001 ) ;
   qsortd ( 0 , bootstrap_reps-1 , q01 ) ;
   qsortd ( 0 , bootstrap_reps-1 , q05 ) ;
   qsortd ( 0 , bootstrap_reps-1 , q10 ) ;

   // Print for user
   fprintf ( fpReport, "\n\nDrawdown approximate bounds." ) ;
   fprintf ( fpReport, "\nRows are drawdown probability, columns are confidence in bounds." ) ;
   fprintf ( fpReport, "\n          0.5       0.6       0.7       0.8       0.9       0.95" ) ;
   fprintf ( fpReport, "\n0.001  %8.3lf  %8.3lf  %8.3lf  %8.3lf  %8.3lf  %8.3lf",
             find_quantile ( bootstrap_reps , q001 , 0.5 ),
             find_quantile ( bootstrap_reps , q001 , 0.6 ),
             find_quantile ( bootstrap_reps , q001 , 0.7 ),
             find_quantile ( bootstrap_reps , q001 , 0.8 ),
             find_quantile ( bootstrap_reps , q001 , 0.9 ),
             find_quantile ( bootstrap_reps , q001 , 0.95 ) ) ;

   fprintf ( fpReport, "\n0.01   %8.3lf  %8.3lf  %8.3lf  %8.3lf  %8.3lf  %8.3lf",
             find_quantile ( bootstrap_reps , q01 , 0.5 ),
             find_quantile ( bootstrap_reps , q01 , 0.6 ),
             find_quantile ( bootstrap_reps , q01 , 0.7 ),
             find_quantile ( bootstrap_reps , q01 , 0.8 ),
             find_quantile ( bootstrap_reps , q01 , 0.9 ),
             find_quantile ( bootstrap_reps , q01 , 0.95 ) ) ;

   fprintf ( fpReport, "\n0.05   %8.3lf  %8.3lf  %8.3lf  %8.3lf  %8.3lf  %8.3lf",
             find_quantile ( bootstrap_reps , q05 , 0.5 ),
             find_quantile ( bootstrap_reps , q05 , 0.6 ),
             find_quantile ( bootstrap_reps , q05 , 0.7 ),
             find_quantile ( bootstrap_reps , q05 , 0.8 ),
             find_quantile ( bootstrap_reps , q05 , 0.9 ),
             find_quantile ( bootstrap_reps , q05 , 0.95 ) ) ;

   fprintf ( fpReport, "\n0.10   %8.3lf  %8.3lf  %8.3lf  %8.3lf  %8.3lf  %8.3lf",
             find_quantile ( bootstrap_reps , q10 , 0.5 ),
             find_quantile ( bootstrap_reps , q10 , 0.6 ),
             find_quantile ( bootstrap_reps , q10 , 0.7 ),
             find_quantile ( bootstrap_reps , q10 , 0.8 ),
             find_quantile ( bootstrap_reps , q10 , 0.9 ),
             find_quantile ( bootstrap_reps , q10 , 0.95 ) ) ;


FINISH:
   if (fpList != NULL)
      fclose ( fpList ) ;

   if (_heapchk() != _HEAPOK) {       // This is optional cheap insurance.  Pull it if you wish.
      fprintf ( fpReport, "\nBad heap!" ) ;
      fclose ( fpReport ) ;
      exit ( 1 ) ;
      }

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
   if (bootsample != NULL)
      free ( bootsample ) ;
   if (quantile_sample != NULL)
      free ( quantile_sample ) ;
   if (work != NULL)
      free ( work ) ;
   if (q001 != NULL)
      free ( q001 ) ;
   if (q01 != NULL)
      free ( q01 ) ;
   if (q05 != NULL)
      free ( q05 ) ;
   if (q10 != NULL)
      free ( q10 ) ;

   fclose ( fpReport ) ;

   exit ( return_value ) ;
}
