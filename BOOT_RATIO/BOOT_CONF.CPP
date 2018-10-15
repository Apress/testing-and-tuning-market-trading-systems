/******************************************************************************/
/*                                                                            */
/*  BOOT_CONF - Assorted bootstrap confidence intervals                       */
/*                                                                            */
/******************************************************************************/

#include <math.h>
#include <stdlib.h>

double unifrand () ;
void qsortd ( int first , int last , double *data ) ;
double normal_cdf ( double z ) ;
double inverse_normal_cdf ( double p ) ;


/*
--------------------------------------------------------------------------------

   boot_conf_pctile - Compute confidence intervals using percentile method

--------------------------------------------------------------------------------
*/

void boot_conf_pctile (
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
   )
{
   int i, rep, k ;

   for (rep=0 ; rep<nboot ; rep++) {    // Do all bootstrap reps (b from 1 to B)

      for (i=0 ; i<n ; i++) {           // Generate the bootstrap sample
         k = (int) (unifrand() * n) ;   // Select a case from the sample
         if (k >= n)                    // Should never happen, but be prepared
            k = n - 1 ;
         xwork[i] = x[k] ;              // Put bootstrap sample in work
         }

      work2[rep] = user_t ( n , xwork ) ;
      }

   qsortd ( 0 , nboot-1 , work2 ) ;     // Sort ascending

   k = (int) (0.025 * (nboot + 1)) - 1 ; // Unbiased quantile estimator
   if (k < 0)
      k = 0 ;
   *low2p5 = work2[k] ;
   *high2p5 = work2[nboot-1-k] ;

   k = (int) (0.05 * (nboot + 1)) - 1 ;
   if (k < 0)
      k = 0 ;
   *low5 = work2[k] ;
   *high5 = work2[nboot-1-k] ;

   k = (int) (0.10 * (nboot + 1)) - 1 ;
   if (k < 0)
      k = 0 ;
   *low10 = work2[k] ;
   *high10 = work2[nboot-1-k] ;
}

/*
--------------------------------------------------------------------------------

   boot_conf_BCa - Compute confidence intervals using improved quantile method

--------------------------------------------------------------------------------
*/

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
   )
{
   int i, rep, k, z0_count ;
   double param, theta_hat, theta_dot, z0, zlo, zhi, alo, ahi ;
   double xtemp, xlast, diff, numer, denom, accel ;

   theta_hat = user_t ( n , x ) ;       // Parameter for full set

   z0_count = 0 ;                       // Will count for computing z0 later

   for (rep=0 ; rep<nboot ; rep++) {    // Do all bootstrap reps (b from 1 to B)

      for (i=0 ; i<n ; i++) {           // Generate the bootstrap sample
         k = (int) (unifrand() * n) ;   // Select a case from the sample
         if (k >= n)                    // Should never happen, but be prepared
            k = n - 1 ;
         xwork[i] = x[k] ;              // Put bootstrap sample in work
         }

      param = user_t ( n , xwork ) ;    // Param for this bootstrap rep
      work2[rep] = param ;              // Save it for CDF later
      if (param < theta_hat)            // Count how many < full set param
         ++z0_count ;                   // For computing z0 later
      }

   if (z0_count >= nboot)               // Prevent nastiness
      z0_count = nboot - 1 ;
   if (z0_count <= 0)
      z0_count = 1 ;

   z0 = inverse_normal_cdf ( (double) z0_count / (double) nboot ) ;

/*
   Do the jackknife for computing accel.
   Borrow xwork for storing jackknifed parameter values.
*/

   xlast = x[n-1] ;
   theta_dot = 0.0 ;
   for (i=0 ; i<n ; i++) {          // Jackknife
      xtemp = x[i] ;                // Preserve case being temporarily removed
      x[i] = xlast ;                // Swap in last case
      param = user_t ( n-1 , x ) ;  // Param for this jackknife
      theta_dot += param ;          // Cumulate mean across jackknife
      xwork[i] = param ;            // Save for computing accel later
      x[i] = xtemp ;                // Restore original case
      }

/*
   Compute accel
*/

   theta_dot /= n ;
   numer = denom = 0.0 ;
   for (i=0 ; i<n ; i++) {
      diff = theta_dot - xwork[i] ;
      xtemp = diff * diff ;
      denom += xtemp ;
      numer += xtemp * diff ;
      }

   denom = sqrt ( denom ) ;
   denom = denom * denom * denom ;
   accel = numer / (6.0 * denom + 1.e-60) ;

/*
   Compute the outputs
*/

   qsortd ( 0 , nboot-1 , work2 ) ;     // Sort ascending

   zlo = inverse_normal_cdf ( 0.025 ) ;
   zhi = inverse_normal_cdf ( 0.975 ) ;
   alo = normal_cdf ( z0 + (z0 + zlo) / (1.0 - accel * (z0 + zlo)) ) ;
   ahi = normal_cdf ( z0 + (z0 + zhi) / (1.0 - accel * (z0 + zhi)) ) ;
   k = (int) (alo * (nboot + 1)) - 1 ; // Unbiased quantile estimator
   if (k < 0)
      k = 0 ;
   *low2p5 = work2[k] ;
   k = (int) ((1.0-ahi) * (nboot + 1)) - 1 ;
   if (k < 0)
      k = 0 ;
   *high2p5 = work2[nboot-1-k] ;

   zlo = inverse_normal_cdf ( 0.05 ) ;
   zhi = inverse_normal_cdf ( 0.95 ) ;
   alo = normal_cdf ( z0 + (z0 + zlo) / (1.0 - accel * (z0 + zlo)) ) ;
   ahi = normal_cdf ( z0 + (z0 + zhi) / (1.0 - accel * (z0 + zhi)) ) ;
   k = (int) (alo * (nboot + 1)) - 1 ; // Unbiased quantile estimator
   if (k < 0)
      k = 0 ;
   *low5 = work2[k] ;
   k = (int) ((1.0-ahi) * (nboot + 1)) - 1 ;
   if (k < 0)
      k = 0 ;
   *high5 = work2[nboot-1-k] ;

   zlo = inverse_normal_cdf ( 0.10 ) ;
   zhi = inverse_normal_cdf ( 0.90 ) ;
   alo = normal_cdf ( z0 + (z0 + zlo) / (1.0 - accel * (z0 + zlo)) ) ;
   ahi = normal_cdf ( z0 + (z0 + zhi) / (1.0 - accel * (z0 + zhi)) ) ;
   k = (int) (alo * (nboot + 1)) - 1 ; // Unbiased quantile estimator
   if (k < 0)
      k = 0 ;
   *low10 = work2[k] ;
   k = (int) ((1.0-ahi) * (nboot + 1)) - 1 ;
   if (k < 0)
      k = 0 ;
   *high10 = work2[nboot-1-k] ;
}