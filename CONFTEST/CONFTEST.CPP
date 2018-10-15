/******************************************************************************/
/*                                                                            */
/*  CONFTEST - Test quantile confidence intervals via incomplete beta         */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>

double unifrand () ;
void qsortd ( int istart , int istop , double *x ) ;
double orderstat_tail ( int n , double q , int m ) ;
double quantile_conf ( int n , int m , double conf ) ;

int main (
   int argc ,    // Number of command line arguments (includes prog name)
   char *argv[]  // Arguments (prog name is argv[0])
   )

{
   int i, itry, nsamps, divisor, lower_bound_index, upper_bound_index ; 
   int lower_bound_fail_above_count, lower_bound_fail_below_count ;
   int upper_bound_fail_above_count, upper_bound_fail_below_count ;
   int lower_bound_low_q_count, lower_bound_high_q_count ;
   int upper_bound_low_q_count, upper_bound_high_q_count ;
   int lower_p_of_q_low_count, lower_p_of_q_high_count ;
   int upper_p_of_q_low_count, upper_p_of_q_high_count ;
   double *x, f, lower_fail_rate, lower_bound, upper_bound, upper_fail_rate ;
   double lower_bound_low_q, lower_bound_high_q, lower_bound_low_theory, lower_bound_high_theory ;
   double upper_bound_low_q, upper_bound_high_q, upper_bound_low_theory, upper_bound_high_theory ;
   double p_of_q, p_of_q_low_q, p_of_q_high_q ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 6) {
      printf ( "\nUsage: CONFTEST  nsamples fail_rate low_q high_q p_of_q" ) ;
      printf ( "\n  nsamples - Number of cases in each trial (at least 20)" ) ;
      printf ( "\n  fail_rate - Desired rate of failure for computed bound (smallish)" ) ;
      printf ( "\n  low_q - Worrisome failure rate below desired (< fail_rate)" ) ;
      printf ( "\n  high_q - Worrisome failure rate above desired (> fail_rate)" ) ;
      printf ( "\n  p_of_q - Small probability of failure; to get limits" ) ;
      exit ( 1 ) ;
      }

   nsamps = atoi ( argv[1] ) ;
   lower_fail_rate = atof ( argv[2] ) ;    // Our desired lower bound's failure rate
   lower_bound_low_q = atof ( argv[3] ) ;  // Test 1:We are unhappy if computed lower bound <= quantile for this
   lower_bound_high_q = atof ( argv[4] ) ; // Or if computed lower bound >= quantile for this
   p_of_q = atof ( argv[5] ) ;             // Test 2: We want this tiny chance of being deceived
#else
   nsamps = 100000 ;
   lower_fail_rate = 0.1 ;       // Our desired lower bound
   lower_bound_low_q = 0.0975 ;  // Test 1:We are unhappy if computed lower bound <= quantile for this
   lower_bound_high_q = 0.101 ;  // Or if computed lower bound >= quantile for this
   p_of_q = 0.01 ;               // Test 2: We want this tiny chance of being deceived
#endif

   if (nsamps < 20  ||  lower_bound_low_q >= lower_fail_rate  ||  lower_bound_high_q <= lower_fail_rate) {
      printf ( "\nUsage: CONFTEST  nsamples fail_rate low_q high_q p_of_q" ) ;
      printf ( "\n  nsamples - Number of cases in each trial (at least 20)" ) ;
      printf ( "\n  fail_rate - Desired rate of failure for computed bound (smallish)" ) ;
      printf ( "\n  low_q - Worrisome failure rate below desired (< fail_rate)" ) ;
      printf ( "\n  high_q - Worrisome failure rate above desired (> fail_rate)" ) ;
      printf ( "\n  p_of_q - Small probability of failure; to get limits" ) ;
      exit ( 1 ) ;
      }

/*
   Allocate memory and initialize
*/

   x = (double *) malloc ( nsamps * sizeof(double) ) ;

   divisor = 1000000 / nsamps ;  // Strictly for progress reporting
   if (divisor < 2)
      divisor = 2 ;

   lower_bound_index = (int) (lower_fail_rate * (nsamps + 1) ) - 1 ;  // Unbiased; C++ index is origin zero
   if (lower_bound_index < 0)
      lower_bound_index = 0 ;

   lower_bound_low_theory = 1.0 - orderstat_tail ( nsamps , lower_bound_low_q , lower_bound_index+1 ) ;
   lower_bound_high_theory = orderstat_tail ( nsamps , lower_bound_high_q , lower_bound_index+1 ) ;

   p_of_q_low_q = quantile_conf ( nsamps , lower_bound_index+1 , 1.0 - p_of_q ) ;
   p_of_q_high_q = quantile_conf ( nsamps , lower_bound_index+1 , p_of_q ) ;

   printf ( "\nnsamps=%d  lower_fail_rate=%.3lf  lower_bound_low_q=%.4lf  p=%.4lf  lower_bound_high_q=%.4lf  p=%.4lf",
            nsamps, lower_fail_rate, lower_bound_low_q, lower_bound_low_theory, lower_bound_high_q, lower_bound_high_theory ) ;

   printf ( "\np_of_q=%.3lf  low_q=%.4lf  high_q=%.4lf", p_of_q, p_of_q_low_q, p_of_q_high_q ) ;

   // Next section is for optional upper bound stuff
   upper_bound_index = nsamps-1-lower_bound_index ;
   upper_fail_rate = lower_fail_rate ;  // Could be different, but choose to make symmetric here
   upper_bound_low_q = 1.0 - lower_bound_high_q ;  // Note reverse symmetry
   upper_bound_high_q = 1.0 - lower_bound_low_q ;
   upper_bound_low_theory = lower_bound_high_theory ;
   upper_bound_high_theory = lower_bound_low_theory ;

/*
   Get ready to go
*/

   printf ( "\n\nPress any key to begin..." ) ;
   _getch () ;

   lower_bound_fail_above_count = lower_bound_fail_below_count = 0 ;
   lower_bound_low_q_count = lower_bound_high_q_count = 0 ;
   lower_p_of_q_low_count = lower_p_of_q_high_count = 0 ;
   upper_bound_fail_above_count = upper_bound_fail_below_count = 0 ;
   upper_bound_low_q_count = upper_bound_high_q_count = 0 ;
   upper_p_of_q_low_count = upper_p_of_q_high_count = 0 ;

/*
   Here we go
*/

   for (itry=1 ; ; itry++) {
      f = 1.0 / itry ;

      if ((itry % divisor) == 1)
         printf ( "\n\n%d", itry ) ;

/*
   Generate this try's data.
   A uniform distribution is convenient because its quantile function is an identity.
*/

      for (i=0 ; i<nsamps ; i++)
         x[i] = unifrand () ;

      qsortd ( 0 , nsamps-1 , x ) ;

      lower_bound = x[lower_bound_index] ;  // This is what we are usually most interested in

/*
   Tally
   Recall that we are using a uniform distribution, whose quantile function is an identity.
   Thus, lower_failure_rate is both the failure rate AND the quantile at this rate.
*/

      if (lower_bound > lower_fail_rate)   // This and the next should fail with about 0.5 probability
         ++lower_bound_fail_above_count ;  // Because lower_bound is unbiased

      if (lower_bound < lower_fail_rate)
         ++lower_bound_fail_below_count ;

      if (lower_bound <= lower_bound_low_q)  // Is our lower bound disturbingly lower than we want?
         ++lower_bound_low_q_count ;

      if (lower_bound >= lower_bound_high_q) // Is our lower bound disturbingly higher than we want?
         ++lower_bound_high_q_count ;

      if (lower_bound <= p_of_q_low_q)  // Ditto, but limits gotten via p of q
         ++lower_p_of_q_low_count ;

      if (lower_bound >= p_of_q_high_q) // Rather than user-specified
         ++lower_p_of_q_high_count ;


      // Next section is for the upper bound

      upper_bound = x[upper_bound_index] ;   // For upper bound test

      if (upper_bound > 1.0-upper_fail_rate) // This and the next should fail with about 0.5 probability
         ++upper_bound_fail_above_count ;    // Because upper_bound is unbiased

      if (upper_bound < 1.0-upper_fail_rate)
         ++upper_bound_fail_below_count ;

      if (upper_bound <= upper_bound_low_q)  // Is our upper bound disturbingly lower than we want?
         ++upper_bound_low_q_count ;

      if (upper_bound >= upper_bound_high_q) // Is our upper bound disturbingly higher than we want?
         ++upper_bound_high_q_count ;

      if (upper_bound <= 1.0-p_of_q_high_q)
         ++upper_p_of_q_low_count ;

      if (upper_bound >= 1.0-p_of_q_low_q)
         ++upper_p_of_q_high_count ;

/*
   Print results so far
*/

      if ((itry % divisor) == 1) {
         printf ( "\n\nLower bound fail above=%5.3lf  Lower bound fail below=%5.3lf",
                   f * lower_bound_fail_above_count, f * lower_bound_fail_below_count ) ;
         printf ( "\nLower bound below lower limit=%5.4lf  theory p=%.4lf  above upper limit=%5.4lf  theory p=%.4lf",
                   f * lower_bound_low_q_count, lower_bound_low_theory, f * lower_bound_high_q_count, lower_bound_high_theory ) ;
         printf ( "\nLower p_of_q below lower limit=%5.4lf  theory p=%.4lf  above upper limit=%5.4lf  theory p=%.4lf",
                   f * lower_p_of_q_low_count, p_of_q, f * lower_p_of_q_high_count, p_of_q ) ;

         printf ( "\n\nUpper bound fail above=%5.3lf  Upper bound fail below=%5.3lf",
                   f * upper_bound_fail_above_count, f * upper_bound_fail_below_count ) ;
         printf ( "\nUpper bound below lower limit=%5.4lf  theory p=%.4lf  above upper limit=%5.4lf  theory p=%.4lf",
                   f * upper_bound_low_q_count, upper_bound_low_theory, f * upper_bound_high_q_count, upper_bound_high_theory ) ;
         printf ( "\nUpper p_of_q below lower limit=%5.4lf  theory p=%.4lf  above upper limit=%5.4lf  theory p=%.4lf",
                   f * upper_p_of_q_low_count, p_of_q, f * upper_p_of_q_high_count, p_of_q ) ;
         }

      if ((itry % 10) == 1) {    // Stop if user presses ESCape
         if (_kbhit ()) {
            if (_getch() == 27)
               break ;
            }
         }

      } // For all tries

   return EXIT_SUCCESS ;
}
