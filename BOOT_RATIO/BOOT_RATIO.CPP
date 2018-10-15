/******************************************************************************/
/*                                                                            */
/*  BOOT_RATIO - Assorted bootstrap confidence intervals for profit factor    */
/*               and Sharpe ratio                                             */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>

void RAND32M_seed ( int iseed ) ;
double unifrand () ;
double normal () ;
void qsortd ( int istart , int istop , double *x ) ;
double normal_cdf ( double z ) ;
double inverse_normal_cdf ( double p ) ;

void boot_conf_pctile ( // Percentile method
   int n ,              // Number of cases in sample
   double *x ,          // Variable in sample
   double (*user_t) (int , double *) , // Compute parameter
   int nboot ,          // Number of bootstrap replications
   double *low2p5 ,     // Output of lower 2.5% bound
   double *high2p5 ,    // Output of upper 2.5% bound
   double *low5 ,       // Output of lower 5% bound
   double *high5 ,      // Output of upper 5% bound
   double *low10 ,      // Output of lower 10% bound
   double *high10 ,     // Output of upper 10% bound
   double *xwork ,      // Work area n long
   double *work2        // Work area nboot long
   ) ;

void boot_conf_BCa (
   int n ,              // Number of cases in sample
   double *x ,          // Variable in sample
   double (*user_t) (int , double * ) , // Compute parameter
   int nboot ,          // Number of bootstrap replications
   double *low2p5 ,     // Output of lower 2.5% bound
   double *high2p5 ,    // Output of upper 2.5% bound
   double *low5 ,       // Output of lower 5% bound
   double *high5 ,      // Output of upper 5% bound
   double *low10 ,      // Output of lower 10% bound
   double *high10 ,     // Output of upper 10% bound
   double *xwork ,      // Work area n long
   double *work2        // Work area nboot long
   ) ;

static int use_log = 1 ;

/*
--------------------------------------------------------------------------------

   Compute the profit factor

--------------------------------------------------------------------------------
*/

double param_pf ( int n , double *x )
{
   int i ;
   double numer, denom, val ;

   numer = denom = 1.e-10 ;  // Prevent either from being zero
   for (i=0 ; i<n ; i++) {
      if (x[i] > 0.0)
         numer += x[i] ;
      else 
         denom -= x[i] ;
      }

   val = numer / denom ;
   return use_log  ?  log(val) : val  ;
}

/*
--------------------------------------------------------------------------------

   Compute the Sharpe ratio (not normalized in any way)

--------------------------------------------------------------------------------
*/

double param_sr ( int n , double *x )
{
   int i ;
   double numer, denom, diff, val ;

   numer = denom = 0.0 ;
   for (i=0 ; i<n ; i++)
      numer += x[i] ;
   numer /= n ;                 // Mean return

   for (i=0 ; i<n ; i++) {
      diff = x[i] - numer ;
      denom += diff * diff ;
      }
   denom = sqrt ( denom / n ) ; // Standard deviation of returns

   if (denom > 0.0)             // Should always be true
      val = numer / denom ;
   else 
      val = 1.e30 ;             // Get user's attention if a problem

   return val  ;
}


/*
--------------------------------------------------------------------------------

   Optional main to test it

   The user specifies a probability of winning, with 0.5 meaning a flat system.
   All trades (win or loss) have a normal distribution with mean $1000
   and standard deviation $200.

--------------------------------------------------------------------------------
*/

int main (
   int argc ,    // Number of command line arguments (includes prog name)
   char *argv[]  // Arguments (prog name is argv[0])
   )

{
   int i, ntries, itry, nsamps, nboot, divisor, ndone ;
   int low2p5, high2p5, low5, high5, low10, high10 ;
   double prob, true_pf, true_sr, *x, *xwork, *work2, *param ;
   double *low2p5_1, *high2p5_1, *low5_1, *high5_1, *low10_1, *high10_1 ;
   double *low2p5_2, *high2p5_2, *low5_2, *high5_2, *low10_2, *high10_2 ;
   double *low2p5_3, *high2p5_3, *low5_3, *high5_3, *low10_3, *high10_3 ;
   double mean_param, true_sum, true_sumsq ;
   char line1[256], line2[256], line3[256], line4[256] ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 5) {
      printf ( "\nUsage: BOOT_RATIO  nsamples  nboot  ntries  prob" ) ;
      printf ( "\n  nsamples - Number of price changes in market history" ) ;
      printf ( "\n  nboot - Number of bootstrap replications" ) ;
      printf ( "\n  ntries - Number of trials for generating summary" ) ;
      printf ( "\n  prob - Probability that a trade will be a win" ) ;
      exit ( 1 ) ;
      }

   nsamps = atoi ( argv[1] ) ;
   nboot = atoi ( argv[2] ) ;
   ntries = atoi ( argv[3] ) ;
   prob = atof ( argv[4] ) ;
#else
   nsamps = 1000 ;
   nboot = 10 ;
   ntries = 1000 ;
   prob = 0.7 ;
#endif

   if ((nsamps <= 0)  ||  (nboot <= 0)  ||  (ntries <= 0)
    || (prob < 0.0)  ||  (prob >= 1.0)) {
      printf (
      "\nUsage: BOOT_RATIO  nsamples  nboot  ntries  prob" ) ;
      exit ( 1 ) ;
      }

   true_pf = prob / (1.0 - prob) ;  // Definition
   if (use_log)
      true_pf = log ( true_pf ) ;

   divisor = 10000000 / (nsamps * nboot) ;  // This is for progress reports only
   if (divisor < 2)
      divisor = 2 ;

/*
   Allocate memory and initialize
*/

   x = (double *) malloc ( nsamps * sizeof(double) ) ;
   xwork = (double *) malloc ( nsamps * sizeof(double) ) ;
   work2 = (double *) malloc ( nboot * sizeof(double) ) ;
   param = (double *) malloc ( ntries * sizeof(double) ) ;
   low2p5_1 = (double *) malloc ( ntries * sizeof(double) ) ;
   high2p5_1 = (double *) malloc ( ntries * sizeof(double) ) ;
   low5_1 = (double *) malloc ( ntries * sizeof(double) ) ;
   high5_1 = (double *) malloc ( ntries * sizeof(double) ) ;
   low10_1 = (double *) malloc ( ntries * sizeof(double) ) ;
   high10_1 = (double *) malloc ( ntries * sizeof(double) ) ;
   low2p5_2 = (double *) malloc ( ntries * sizeof(double) ) ;
   high2p5_2 = (double *) malloc ( ntries * sizeof(double) ) ;
   low5_2 = (double *) malloc ( ntries * sizeof(double) ) ;
   high5_2 = (double *) malloc ( ntries * sizeof(double) ) ;
   low10_2 = (double *) malloc ( ntries * sizeof(double) ) ;
   high10_2 = (double *) malloc ( ntries * sizeof(double) ) ;
   low2p5_3 = (double *) malloc ( ntries * sizeof(double) ) ;
   high2p5_3 = (double *) malloc ( ntries * sizeof(double) ) ;
   low5_3 = (double *) malloc ( ntries * sizeof(double) ) ;
   high5_3 = (double *) malloc ( ntries * sizeof(double) ) ;
   low10_3 = (double *) malloc ( ntries * sizeof(double) ) ;
   high10_3 = (double *) malloc ( ntries * sizeof(double) ) ;

/*
   Main outer loop does all tries for profit factor
*/

   true_sum = true_sumsq = 0.0 ;
   for (itry=0 ; itry<ntries ; itry++) {

      if ((itry % divisor) == 0)
         printf ( "\n\n\nTry %d", itry ) ;

      RAND32M_seed ( itry + (itry << 16) ) ; // Ensure same data for profit factor & Sharpe Ratio
      for (i=0 ; i<nsamps ; i++) {
         x[i] = 0.01 + 0.002 * normal() ;  // Generate a trade amount
         if (unifrand() > prob)
            x[i] = -x[i] ;       // Make some of the trades into losses
         true_sum += x[i] ;      // Cumulate for true_pf in second set of tests
         true_sumsq += x[i] * x[i] ;
         }

      param[itry] = param_pf ( nsamps , x  ) ;

      boot_conf_pctile ( nsamps , x , param_pf , nboot ,
                   &low2p5_1[itry] , &high2p5_1[itry] , &low5_1[itry] , &high5_1[itry] , 
                   &low10_1[itry] , &high10_1[itry] , xwork , work2 ) ;

      boot_conf_BCa ( nsamps , x , param_pf , nboot ,
           &low2p5_2[itry] , &high2p5_2[itry] , &low5_2[itry] , &high5_2[itry] , 
           &low10_2[itry] , &high10_2[itry] , xwork , work2 ) ;

      // The inverted pivot intervals are trivially obtained from the
      // percentile intervals
      low2p5_3[itry] = 2.0 * param[itry] - high2p5_1[itry] ;
      high2p5_3[itry] = 2.0 * param[itry] - low2p5_1[itry] ;
      low5_3[itry] = 2.0 * param[itry] - high5_1[itry] ;
      high5_3[itry] = 2.0 * param[itry] - low5_1[itry] ;
      low10_3[itry] = 2.0 * param[itry] - high10_1[itry] ;
      high10_3[itry] = 2.0 * param[itry] - low10_1[itry] ;

/*
   Occasionally, to let the user know that computation is continuing,
   tally results so far and print them.  We also do this on the last try.
   Write these results in line1-line4 so that they can be printed in a summary
   table at the end of the program.
*/

      if (((itry % divisor) == 1)
       || (itry == ntries-1) ) {      // Don't do this every try!  Too slow.
         ndone = itry + 1 ;           // This many tries done (and in arrays)

         mean_param = 0.0 ;
         for (i=0 ; i<ndone ; i++)
            mean_param += param[i] ;
         mean_param /= ndone ;

         if (use_log)
            sprintf_s ( line1, "Mean log pf = %.5lf true = %.5lf", mean_param, true_pf ) ;
         else
            sprintf_s ( line1, "Mean pf = %.5lf true = %.5lf", mean_param, true_pf ) ;
         printf ( "\n%s", line1 ) ;

/*
   Process test 1 of 3 (Percentile method)
*/

         low2p5 = high2p5 = low5 = high5 = low10 = high10 = 0 ;
         for (i=0 ; i<ndone ; i++) {
            if (low2p5_1[i] > true_pf)
               ++low2p5 ;
            if (high2p5_1[i] < true_pf)
               ++high2p5 ;
            if (low5_1[i] > true_pf)
               ++low5 ;
            if (high5_1[i] < true_pf)
               ++high5 ;
            if (low10_1[i] > true_pf)
               ++low10 ;
            if (high10_1[i] < true_pf)
               ++high10 ;
            }
         sprintf_s ( line2,
            "Pctile 2.5: (%4.2lf %4.2lf)  5: (%4.2lf %4.2lf)  10: (%5.2lf %5.2lf)",
            100.0 * low2p5 / ndone , 100.0 * high2p5 / ndone ,
            100.0 * low5 / ndone , 100.0 * high5 / ndone ,
            100.0 * low10 / ndone , 100.0 * high10 / ndone ) ;
         printf ( "\n%s", line2 ) ;

/*
   Process test 2 of 3 (BCa method)
*/

         low2p5 = high2p5 = low5 = high5 = low10 = high10 = 0 ;
         for (i=0 ; i<ndone ; i++) {
            if (low2p5_2[i] > true_pf)
               ++low2p5 ;
            if (high2p5_2[i] < true_pf)
               ++high2p5 ;
            if (low5_2[i] > true_pf)
               ++low5 ;
            if (high5_2[i] < true_pf)
               ++high5 ;
            if (low10_2[i] > true_pf)
               ++low10 ;
            if (high10_2[i] < true_pf)
               ++high10 ;
            }
         sprintf_s ( line3,
            "BCa    2.5: (%4.2lf %4.2lf)  5: (%4.2lf %4.2lf)  10: (%5.2lf %5.2lf)",
            100.0 * low2p5 / ndone , 100.0 * high2p5 / ndone ,
            100.0 * low5 / ndone , 100.0 * high5 / ndone ,
            100.0 * low10 / ndone , 100.0 * high10 / ndone ) ;
         printf ( "\n%s", line3 ) ;

/*
   Process test 3 of 3 (Pivot method)
*/

         low2p5 = high2p5 = low5 = high5 = low10 = high10 = 0 ;
         for (i=0 ; i<ndone ; i++) {
            if (low2p5_3[i] > true_pf)
               ++low2p5 ;
            if (high2p5_3[i] < true_pf)
               ++high2p5 ;
            if (low5_3[i] > true_pf)
               ++low5 ;
            if (high5_3[i] < true_pf)
               ++high5 ;
            if (low10_3[i] > true_pf)
               ++low10 ;
            if (high10_3[i] < true_pf)
               ++high10 ;
            }
         sprintf_s ( line4,
            "Pivot  2.5: (%4.2lf %4.2lf)  5: (%4.2lf %4.2lf)  10: (%5.2lf %5.2lf)",
            100.0 * low2p5 / ndone , 100.0 * high2p5 / ndone ,
            100.0 * low5 / ndone , 100.0 * high5 / ndone ,
            100.0 * low10 / ndone , 100.0 * high10 / ndone ) ;
         printf ( "\n%s", line4 ) ;
         } // If progress report

      if ((itry % 10) == 1) {    // This lets the user press ESCape to abort computation
         if (_kbhit ()) {
            if (_getch() == 27)
               break ;
            }
         }

      } // For all tries for profit factor

/*
--------------------------------------------------------------------------------

   Profit factor is complete.  Now do Sharpe Ratio.
   We have to do this after pf is complete because we use the pf trades
   to compute the true Sharpe ratio.

--------------------------------------------------------------------------------
*/

   true_sum /= (ntries * nsamps) ;                          // Mean return
   true_sumsq /= (ntries * nsamps) ;
   true_sumsq = sqrt ( true_sumsq - true_sum * true_sum ) ; // StdDev of returns
   true_sr = true_sum / true_sumsq ;

/*
   Main outer loop does all tries for Sharpe ratio
*/

   for (itry=0 ; itry<ntries ; itry++) {

      if ((itry % divisor) == 0)
         printf ( "\n\n\nTry %d", itry ) ;

      RAND32M_seed ( itry + (itry << 16) ) ; // Ensure same data for profit factor & Sharpe Ratio
      for (i=0 ; i<nsamps ; i++) {
         x[i] = 0.01 + 0.002 * normal() ;  // Generate a trade amount
         if (unifrand() > prob)
            x[i] = -x[i] ;       // Make some of the trades into losses
         }

      param[itry] = param_sr ( nsamps , x  ) ;

      boot_conf_pctile ( nsamps , x , param_sr , nboot ,
                   &low2p5_1[itry] , &high2p5_1[itry] , &low5_1[itry] , &high5_1[itry] , 
                   &low10_1[itry] , &high10_1[itry] , xwork , work2 ) ;

      boot_conf_BCa ( nsamps , x , param_sr , nboot ,
           &low2p5_2[itry] , &high2p5_2[itry] , &low5_2[itry] , &high5_2[itry] , 
           &low10_2[itry] , &high10_2[itry] , xwork , work2 ) ;

      // The inverted pivot intervals are trivially obtained from the
      // percentile intervals
      low2p5_3[itry] = 2.0 * param[itry] - high2p5_1[itry] ;
      high2p5_3[itry] = 2.0 * param[itry] - low2p5_1[itry] ;
      low5_3[itry] = 2.0 * param[itry] - high5_1[itry] ;
      high5_3[itry] = 2.0 * param[itry] - low5_1[itry] ;
      low10_3[itry] = 2.0 * param[itry] - high10_1[itry] ;
      high10_3[itry] = 2.0 * param[itry] - low10_1[itry] ;

      if (((itry % divisor) == 1)
       || (itry == ntries-1) ) {      // Don't do this every try!  Too slow.
         if (itry == ntries-1)
            printf ( "\n\nFinal Sharpe ratio..." ) ;
         ndone = itry + 1 ;           // This many tries done (and in arrays)

         mean_param = 0.0 ;
         for (i=0 ; i<ndone ; i++)
            mean_param += param[i] ;
         mean_param /= ndone ;

         printf ( "\nMean sr = %.5lf  true = %.5lf", mean_param, true_sr ) ;

/*
   Process test 1 of 3
*/

         low2p5 = high2p5 = low5 = high5 = low10 = high10 = 0 ;
         for (i=0 ; i<ndone ; i++) {
            if (low2p5_1[i] > true_sr)
               ++low2p5 ;
            if (high2p5_1[i] < true_sr)
               ++high2p5 ;
            if (low5_1[i] > true_sr)
               ++low5 ;
            if (high5_1[i] < true_sr)
               ++high5 ;
            if (low10_1[i] > true_sr)
               ++low10 ;
            if (high10_1[i] < true_sr)
               ++high10 ;
            }
         printf (
            "\nPctile 2.5: (%4.2lf %4.2lf)  5: (%4.2lf %4.2lf)  10: (%5.2lf %5.2lf)",
            100.0 * low2p5 / ndone , 100.0 * high2p5 / ndone ,
            100.0 * low5 / ndone , 100.0 * high5 / ndone ,
            100.0 * low10 / ndone , 100.0 * high10 / ndone ) ;

/*
   Process test 2 of 3
*/

         low2p5 = high2p5 = low5 = high5 = low10 = high10 = 0 ;
         for (i=0 ; i<ndone ; i++) {
            if (low2p5_2[i] > true_sr)
               ++low2p5 ;
            if (high2p5_2[i] < true_sr)
               ++high2p5 ;
            if (low5_2[i] > true_sr)
               ++low5 ;
            if (high5_2[i] < true_sr)
               ++high5 ;
            if (low10_2[i] > true_sr)
               ++low10 ;
            if (high10_2[i] < true_sr)
               ++high10 ;
            }
         printf (
            "\nBCa    2.5: (%4.2lf %4.2lf)  5: (%4.2lf %4.2lf)  10: (%5.2lf %5.2lf)",
            100.0 * low2p5 / ndone , 100.0 * high2p5 / ndone ,
            100.0 * low5 / ndone , 100.0 * high5 / ndone ,
            100.0 * low10 / ndone , 100.0 * high10 / ndone ) ;

/*
   Process test 3 of 3
*/

         low2p5 = high2p5 = low5 = high5 = low10 = high10 = 0 ;
         for (i=0 ; i<ndone ; i++) {
            if (low2p5_3[i] > true_sr)
               ++low2p5 ;
            if (high2p5_3[i] < true_sr)
               ++high2p5 ;
            if (low5_3[i] > true_sr)
               ++low5 ;
            if (high5_3[i] < true_sr)
               ++high5 ;
            if (low10_3[i] > true_sr)
               ++low10 ;
            if (high10_3[i] < true_sr)
               ++high10 ;
            }
         printf (
            "\nPivot  2.5: (%4.2lf %4.2lf)  5: (%4.2lf %4.2lf)  10: (%5.2lf %5.2lf)",
            100.0 * low2p5 / ndone , 100.0 * high2p5 / ndone ,
            100.0 * low5 / ndone , 100.0 * high5 / ndone ,
            100.0 * low10 / ndone , 100.0 * high10 / ndone ) ;
         } // If progress report

      if ((itry % 10) == 1) {
         if (_kbhit ()) {
            if (_getch() == 27)
               break ;
            }
         }

      } // For all tries

/*
   We just printed the results for the Sharpe ratio.
   Now print the results from the profit factor that we saved earlier.
*/

   printf ( "\n\nFinal profit factor..." ) ;
   printf ( "\n%s", line1 ) ;
   printf ( "\n%s", line2 ) ;
   printf ( "\n%s", line3 ) ;
   printf ( "\n%s", line4 ) ;

   printf ( "\n\nnsamps=%d  nboot=%d  ntries=%d  prob=%.3lf",
            nsamps, nboot, ntries, prob ) ;
   printf ( "\nPress any key..." ) ;
   _getch () ;
   return EXIT_SUCCESS ;
}
