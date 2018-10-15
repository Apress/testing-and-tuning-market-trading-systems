/*******************************************************************************/
/*                                                                             */
/*  Drawdown - Test several methods for finding percentiles of future drawdown */
/*                                                                             */
/*******************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>

#define PI 3.141592653589793
#define POP_MULT 1000

double unifrand () ;
void qsortd ( int first , int last , double *data ) ;


/*
--------------------------------------------------------------------------------

   Generate a standard normal random variable or a pair of them
   using the Box-Muller method.

--------------------------------------------------------------------------------
*/

double normal ()
{
   double x1, x2 ;

   for (;;) {
      x1 = unifrand () ;
      if (x1 <= 0.0)      // Safety: log(0) is undefined
         continue ;
      x1 = sqrt ( -2.0 * log ( x1 )) ;
      x2 = cos ( 2.0 * PI * unifrand () ) ;
      return x1 * x2 ;
      }
}


/*
--------------------------------------------------------------------------------

   Generate a set of trades

--------------------------------------------------------------------------------
*/

void get_trades (
   int n_changes ,      // Number of price changes (available history)
   int n_trades ,       // Number of these changes defining drawdown period
   double win_prob ,    // Probability 0-1 of a winning trade
   int make_changes ,   // Draw a new random sample from which bootstraps are drawn?
   double *changes ,    // Work area for storing n_changes changes
   double *trades       // n_trades are returned here
   )
{
   int i, k, itrade ;

   if (make_changes) {   // Generate the sample?
      for (i=0 ; i<n_changes ; i++) {
         changes[i] = normal () ;
         if (unifrand() < win_prob)
            changes[i] = fabs ( changes[i] ) ;
         else
            changes[i] = -fabs ( changes[i] ) ;
         }
      }

   // Get the trades from a standard bootstrap
   for (itrade=0 ; itrade<n_trades ; itrade++) {
      k = (int) (unifrand() * n_changes) ;
      if (k >= n_changes)
         k = n_changes - 1 ;
      trades[itrade] = changes[k] ;
      }
}


/*
--------------------------------------------------------------------------------

   Compute mean return

--------------------------------------------------------------------------------
*/

double mean_return (
   int n ,           // Number of trades
   double *trades    // They are here
   )
{
   int icase ;
   double sum ;

   sum = 0.0 ;

   for (icase=0 ; icase<n ; icase++)
      sum += trades[icase] ;

   return sum / n ;
}


/*
--------------------------------------------------------------------------------

   Compute drawdown

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

   return dd ;
}


/*
--------------------------------------------------------------------------------

   Compute four drawdown quantiles

--------------------------------------------------------------------------------
*/

void drawdown_quantiles (
   int n_changes ,       // Number of price changes (available history)
   int n_trades ,        // Number of trades
   double *b_changes ,   // n_changes bootstrap sample changes supplied here
   int nboot ,           // Number of bootstraps used to compute quantiles
   double *bootsample ,  // Work area n_trades long
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
         bootsample[i] = b_changes[k] ;
         }
      work[iboot] = drawdown ( n_trades , bootsample ) ;
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

   Main routine is here

--------------------------------------------------------------------------------
*/


int main (
   int argc ,    // Number of command line arguments (includes prog name)
   char *argv[]  // Arguments (prog name is argv[0])
   )

{
   int i, itest, iboot, ipop, n_changes, n_trades, bootstrap_reps, quantile_reps, test_reps, make_changes ;
   int count_incorrect_meanret_001, count_incorrect_meanret_01, count_incorrect_meanret_05, count_incorrect_meanret_10 ;
   int count_incorrect_drawdown_001, count_incorrect_drawdown_01, count_incorrect_drawdown_05, count_incorrect_drawdown_10 ;
   int count_correct_001, count_correct_01, count_correct_05, count_correct_10 ;
   double crit, win_prob, *changes, *bootsample, *trades, bound_conf ;
   double *incorrect_meanrets, *incorrect_drawdowns ;
   double incorrect_meanret_001, incorrect_meanret_01, incorrect_meanret_05, incorrect_meanret_10 ;
   double incorrect_drawdown_001, incorrect_drawdown_01, incorrect_drawdown_05, incorrect_drawdown_10 ;
   double *correct_q001, *correct_q01, *correct_q05, *correct_q10, *work ;
   double correct_q001_bound, correct_q01_bound, correct_q05_bound, correct_q10_bound ;
   FILE *fp ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 8) {
      printf ( "\nUsage: DRAWDOWN  Nchanges  Ntrades  WinProb  BoundConf  BootstrapReps  QuantileReps  TestReps" ) ;
      printf ( "\n  Nchanges - Number of price changes" ) ;
      printf ( "\n  Ntrades - Number of trades" ) ;
      printf ( "\n  WinProb - Probability of winning" ) ;
      printf ( "\n  BoundConf - Confidence (typically .5-.999) in correct dd bound" ) ;
      printf ( "\n  BootstrapReps - Number of bootstrap reps" ) ;
      printf ( "\n  QuantileReps - Number of bootstrap reps for finding drawdown quantiles" ) ;
      printf ( "\n  TestReps - Number of testing reps for this study" ) ;
      exit ( 1 ) ;
      }

   n_changes = atoi ( argv[1] ) ;
   n_trades = atoi ( argv[2] ) ;
   win_prob = atof ( argv[3] ) ;
   bound_conf = atof ( argv[4] ) ;
   bootstrap_reps = atoi ( argv[5] ) ;
   quantile_reps = atoi ( argv[6] ) ;
   test_reps = atoi ( argv[7] ) ;
#else
   n_changes = 252 ;
   n_trades = 252 ;
   win_prob = 0.5 ;
   bound_conf = 0.8 ;
   bootstrap_reps = 1000 ;
   quantile_reps = 1000 ;
   test_reps = 1000 ;
#endif

   if (n_changes < 2) {
      printf ( "\nERROR... Nchanges must be at least 2" ) ;
      return EXIT_FAILURE ;
      }

   if (n_trades < 2) {
      printf ( "\nERROR... Ntrades must be at least 2" ) ;
      return EXIT_FAILURE ;
      }

   if (n_trades > n_changes) {
      printf ( "\nERROR... Ntrades must not exceed Nchanges" ) ;
      return EXIT_FAILURE ;
      }

   if (win_prob < 0.0  ||  win_prob > 1.0) {
      printf ( "\nERROR... Winning probability must be 0-1" ) ;
      return EXIT_FAILURE ;
      }

   if (bootstrap_reps < 10) {
      printf ( "\nERROR... BootstrapReps must be at least 10" ) ;
      return EXIT_FAILURE ;
      }

   if (quantile_reps < 10) {
      printf ( "\nERROR... QuantileReps must be at least 10" ) ;
      return EXIT_FAILURE ;
      }

   if (test_reps < 1) {
      printf ( "\nERROR... TestReps must be at least 1" ) ;
      return EXIT_FAILURE ;
      }


/*
   Open the text file to which results will be written
   and print user's parameters
*/

   if (fopen_s ( &fp , "DRAWDOWN.LOG" , "wt" )) {
      printf ( "\nCannot open DRAWDOWN.LOG file for writing!" ) ;
      return EXIT_FAILURE ;
      }

   fprintf ( fp, "\nChanges = %d", n_changes ) ;
   fprintf ( fp, "\nTrades = %d", n_trades ) ;
   fprintf ( fp, "\nWin probability = %.4lf", win_prob ) ;
   fprintf ( fp, "\nDD bound confidence = %.4lf", bound_conf ) ;
   fprintf ( fp, "\nBootstrap reps = %d", bootstrap_reps ) ;
   fprintf ( fp, "\nQuantile reps = %d", quantile_reps ) ;
   fprintf ( fp, "\nTest reps = %d", test_reps ) ;


/*
   Allocate memory
*/

   changes = (double *) malloc ( n_changes * sizeof(double) ) ;
   bootsample = (double *) malloc ( n_trades * sizeof(double) ) ;
   trades = (double *) malloc ( n_changes * sizeof(double) ) ;    // Correct test does a bootstrap of all changes
   incorrect_meanrets = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   incorrect_drawdowns = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   correct_q001 = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   correct_q01 = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   correct_q05 = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   correct_q10 = (double *) malloc ( bootstrap_reps * sizeof(double) ) ;
   work = (double *) malloc ( quantile_reps * sizeof(double) ) ;

/*
   Outer (test) loop
*/

   count_incorrect_meanret_001 = count_incorrect_meanret_01 = count_incorrect_meanret_05 = count_incorrect_meanret_10 = 0 ;
   count_incorrect_drawdown_001 = count_incorrect_drawdown_01 = count_incorrect_drawdown_05 = count_incorrect_drawdown_10 = 0 ;
   count_correct_001 = count_correct_01 = count_correct_05 = count_correct_10 = 0 ;

   for (itest=1 ; itest<=test_reps ; itest++) {

/*
   Incorrect method test
*/

      for (iboot=0 ; iboot<bootstrap_reps ; iboot++) {
         make_changes = (iboot == 0)  ?  1 : 0 ; // Generate sample on first pass only
         get_trades ( n_changes , n_trades , win_prob , make_changes , changes , trades ) ;
         incorrect_meanrets[iboot] = mean_return ( n_trades , trades ) ;
         incorrect_drawdowns[iboot] = drawdown ( n_trades , trades ) ;
         } // End of incorrect method bootstrap loop

      // Sort for CDF and find quantiles
      qsortd ( 0 , bootstrap_reps-1 , incorrect_meanrets ) ;
      incorrect_meanret_001 = find_quantile ( bootstrap_reps , incorrect_meanrets , 0.001 ) ;
      incorrect_meanret_01 =  find_quantile ( bootstrap_reps , incorrect_meanrets , 0.01 ) ;
      incorrect_meanret_05 =  find_quantile ( bootstrap_reps , incorrect_meanrets , 0.05 ) ;
      incorrect_meanret_10 =  find_quantile ( bootstrap_reps , incorrect_meanrets , 0.1 ) ;

      qsortd ( 0 , bootstrap_reps-1 , incorrect_drawdowns ) ;
      incorrect_drawdown_001 = find_quantile ( bootstrap_reps , incorrect_drawdowns , 0.999 ) ;
      incorrect_drawdown_01 =  find_quantile ( bootstrap_reps , incorrect_drawdowns , 0.99 ) ;
      incorrect_drawdown_05 =  find_quantile ( bootstrap_reps , incorrect_drawdowns , 0.95 ) ;
      incorrect_drawdown_10 =  find_quantile ( bootstrap_reps , incorrect_drawdowns , 0.9 ) ;

/*
   Correct method test
*/

      for (iboot=0 ; iboot<bootstrap_reps ; iboot++) {
         make_changes = (iboot == 0)  ?  1 : 0 ; // Generate sample on first pass only
         get_trades ( n_changes , n_changes , win_prob , make_changes , changes , trades ) ;
         drawdown_quantiles ( n_changes , n_trades , trades , quantile_reps , bootsample , work ,
                              &correct_q001[iboot] , &correct_q01[iboot] ,&correct_q05[iboot] ,&correct_q10[iboot] ) ;
         } // End of correct method bootstrap loop

      // Sort for CDF and find quantiles
      qsortd ( 0 , bootstrap_reps-1 , correct_q001 ) ;
      qsortd ( 0 , bootstrap_reps-1 , correct_q01 ) ;
      qsortd ( 0 , bootstrap_reps-1 , correct_q05 ) ;
      qsortd ( 0 , bootstrap_reps-1 , correct_q10 ) ;
      correct_q001_bound = find_quantile ( bootstrap_reps , correct_q001 , 1.0 - (1.0 - bound_conf) / 2.0 ) ;
      correct_q01_bound = find_quantile ( bootstrap_reps , correct_q01 , 1.0 - (1.0 - bound_conf) / 2.0 ) ;
      correct_q05_bound = find_quantile ( bootstrap_reps , correct_q05 , bound_conf ) ;
      correct_q10_bound = find_quantile ( bootstrap_reps , correct_q10 , bound_conf ) ;

#if 0
      printf ( "\n\nq001 .5 .9 .95 .99 = %.1lf %.1lf %.1lf %.1lf", 
               find_quantile ( bootstrap_reps , correct_q001 , 0.5 ),
               find_quantile ( bootstrap_reps , correct_q001 , 0.9 ),
               find_quantile ( bootstrap_reps , correct_q001 , 0.95 ),
               find_quantile ( bootstrap_reps , correct_q001 , 0.99 )) ;
      printf ( "\n\nq10 .5 .9 .95 .99 = %.1lf %.1lf %.1lf %.1lf", 
               find_quantile ( bootstrap_reps , correct_q10 , 0.5 ),
               find_quantile ( bootstrap_reps , correct_q10 , 0.9 ),
               find_quantile ( bootstrap_reps , correct_q10 , 0.95 ),
               find_quantile ( bootstrap_reps , correct_q10 , 0.99 )) ;
#endif

/*
   Population test
*/

      for (ipop=0 ; ipop<POP_MULT ; ipop++) {

         for (i=0 ; i<n_trades ; i++) {
            trades[i] = normal () ;
            if (unifrand() < win_prob)
               trades[i] = fabs ( trades[i] ) ;
            else
               trades[i] = -fabs ( trades[i] ) ;
            }

         //-----------------------------------------
         // Compute and test mean return being worse
         //-----------------------------------------

         crit = mean_return ( n_trades , trades ) ;

         if (crit < incorrect_meanret_001)
            ++count_incorrect_meanret_001 ;
         if (crit < incorrect_meanret_01)
            ++count_incorrect_meanret_01 ;
         if (crit < incorrect_meanret_05)
            ++count_incorrect_meanret_05 ;
         if (crit < incorrect_meanret_10)
            ++count_incorrect_meanret_10 ;


         //---------------------------------------------------------
         // Compute and test drawdown being worse (incorrect method)
         //---------------------------------------------------------

         crit = drawdown ( n_trades , trades ) ;

         if (crit > incorrect_drawdown_001)
            ++count_incorrect_drawdown_001 ;
         if (crit > incorrect_drawdown_01)
            ++count_incorrect_drawdown_01 ;
         if (crit > incorrect_drawdown_05)
            ++count_incorrect_drawdown_05 ;
         if (crit > incorrect_drawdown_10)
            ++count_incorrect_drawdown_10 ;


         //---------------------------------------------------------
         // Compute and test drawdown being worse (correct method)
         //---------------------------------------------------------

         if (crit > correct_q001_bound)
            ++count_correct_001 ;

         if (crit > correct_q01_bound)
            ++count_correct_01 ;

         if (crit > correct_q05_bound)
            ++count_correct_05 ;

         if (crit > correct_q10_bound)
            ++count_correct_10 ;

         } // For ipop


/*
   For user's continuing edification, print to the screen the results that we have
   (counts of being worse)
*/

      printf ( "\n\n%d", itest ) ;
      printf ( "\nMean return" ) ;
      printf ( "\n  Actual    Incorrect" ) ;
      printf ( "\n   0.001   %8.5lf", (double) count_incorrect_meanret_001 / (POP_MULT * itest) ) ;
      printf ( "\n   0.01    %8.5lf", (double) count_incorrect_meanret_01  / (POP_MULT * itest) ) ;
      printf ( "\n   0.05    %8.5lf", (double) count_incorrect_meanret_05  / (POP_MULT * itest) ) ;
      printf ( "\n   0.1     %8.5lf", (double) count_incorrect_meanret_10  / (POP_MULT * itest) ) ;

      printf ( "\n\nDrawdown" ) ;
      printf ( "\n  Actual    Incorrect  Correct" ) ;
      printf ( "\n   0.001   %8.5lf  %8.5lf",
               (double) count_incorrect_drawdown_001 / (POP_MULT * itest),
               (double) count_correct_001 / (POP_MULT * itest) ) ;
      printf ( "\n   0.01    %8.5lf  %8.5lf",
               (double) count_incorrect_drawdown_01  / (POP_MULT * itest),
               (double) count_correct_01 / (POP_MULT * itest) ) ;
      printf ( "\n   0.05    %8.5lf  %8.5lf",
               (double) count_incorrect_drawdown_05  / (POP_MULT * itest),
               (double) count_correct_05 / (POP_MULT * itest) ) ;
      printf ( "\n   0.1     %8.5lf  %8.5lf",
               (double) count_incorrect_drawdown_10  / (POP_MULT * itest),
               (double) count_correct_10 / (POP_MULT * itest) ) ;


/*
   Write results to file.
*/

      if (itest % 100 == 0  ||  itest == test_reps  ||  _kbhit()) {
         fprintf ( fp , "\n\n\n" ) ;
         fprintf ( fp , "\nMean return worse (Ratio)" ) ;
         fprintf ( fp , "\n  Actual       Incorrect" ) ;
         fprintf ( fp , "\n   0.001   %8.5lf (%6.2lf)",
                   (double) count_incorrect_meanret_001 / (POP_MULT * itest),
                   ((double) count_incorrect_meanret_001 / (POP_MULT * itest)) / 0.001) ;
         fprintf ( fp , "\n   0.01    %8.5lf (%6.2lf)",
                   (double) count_incorrect_meanret_01  / (POP_MULT * itest),
                   ((double) count_incorrect_meanret_01  / (POP_MULT * itest)) / 0.01) ;
         fprintf ( fp , "\n   0.05    %8.5lf (%6.2lf)",
                   (double) count_incorrect_meanret_05  / (POP_MULT * itest),
                   ((double) count_incorrect_meanret_05  / (POP_MULT * itest)) / 0.05) ;
         fprintf ( fp , "\n   0.1     %8.5lf (%6.2lf)",
                   (double) count_incorrect_meanret_10  / (POP_MULT * itest),
                   ((double) count_incorrect_meanret_10  / (POP_MULT * itest)) / 0.1) ;

         fprintf ( fp , "\n\nDrawdown worse (Ratio)" ) ;
         fprintf ( fp , "\n  Actual     Incorrect          Correct" ) ;
         fprintf ( fp , "\n   0.001   %8.5lf (%6.2lf)  %8.5lf (%6.2lf)",
                   (double) count_incorrect_drawdown_001 / (POP_MULT * itest),
                   ((double) count_incorrect_drawdown_001 / (POP_MULT * itest)) / 0.001,
                   (double) count_correct_001 / (POP_MULT * itest),
                   ((double) count_correct_001 / (POP_MULT * itest)) / 0.001) ;
         fprintf ( fp , "\n   0.01    %8.5lf (%6.2lf)  %8.5lf (%6.2lf)",
                   (double) count_incorrect_drawdown_01  / (POP_MULT * itest),
                   ((double) count_incorrect_drawdown_01  / (POP_MULT * itest)) / 0.01,
                   (double) count_correct_01 / (POP_MULT * itest),
                   ((double) count_correct_01 / (POP_MULT * itest)) / 0.01) ;
         fprintf ( fp , "\n   0.05    %8.5lf (%6.2lf)  %8.5lf (%6.2lf)",
                   (double) count_incorrect_drawdown_05  / (POP_MULT * itest),
                   ((double) count_incorrect_drawdown_05  / (POP_MULT * itest)) / 0.05,
                   (double) count_correct_05 / (POP_MULT * itest),
                   ((double) count_correct_05 / (POP_MULT * itest)) / 0.05) ;
         fprintf ( fp , "\n   0.1     %8.5lf (%6.2lf)  %8.5lf (%6.2lf)",
                   (double) count_incorrect_drawdown_10  / (POP_MULT * itest),
                   ((double) count_incorrect_drawdown_10  / (POP_MULT * itest)) / 0.1,
                   (double) count_correct_10 / (POP_MULT * itest),
                   ((double) count_correct_10 / (POP_MULT * itest)) / 0.10) ;
         }

     if (_kbhit ()) {          // Has the user pressed a key?
         if (_getch() == 27)   // The ESCape key?
            break ;
         }
      } // End of test loop


/*
   Finished
*/

   fclose ( fp ) ;
   free ( changes ) ;
   free ( bootsample ) ;
   free ( trades ) ;
   free ( incorrect_meanrets ) ;
   free ( incorrect_drawdowns ) ;
   free ( correct_q001 ) ;
   free ( correct_q01 ) ;
   free ( correct_q05 ) ;
   free ( correct_q10 ) ;
   free ( work ) ;
   return EXIT_SUCCESS ;
}
