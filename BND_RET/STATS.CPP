/******************************************************************************/
/*                                                                            */
/* Assorted numerical approximations and tests:                               */
/*   Normal CDF   Accurate to 7.5 e-8                                         */
/*   Inverse normal CDF Accurate to 4.5 e-4                                   */
/*   Complementary error function                                             */
/*   Half-normal CDF == CDF of |Sn| / sqrt(n)                                 */
/*   Gamma ( int two_K ) - Gamma distribution for twice argument is an integer*/
/*   Log Gamma (ACM algorithm 291) seems good to about 9-10 significant digits*/
/*   Incomplete Gamma (From Press et al) (Chisq CDF is igamma(df/2,chisq/2)   */
/*   Incomplete Beta (ACM algorithm 179 with modifications through 1976)      */
/*   Inverse Incomplete Beta (based on Jstat algorithm)                       */
/*   Student's t CDF                                                          */
/*   Inverse Student's t CDF                                                  */
/*   F CDF                                                                    */
/*   Poisson pdf                                                              */
/*   Anderson-Darling CDF                                                     */
/*   K-S CDF                                                                  */
/*   Inverse K-S CDF                                                          */
/*   t test (for one sample)                                                  */
/*   t test (for independent samples)                                         */
/*   Mann-Whitney U test                                                      */
/*   K-S test                                                                 */
/*   A-D test                                                                 */
/*   One-way ANOVA                                                            */
/*   Kruskal-Wallis test (nonparametric rank-based ANOVA)                     */
/*   Chi-square test                                                          */
/*   nominal_lambda - Compute Lambda for a pair of nominal variables          */
/*   uncertainty reduction for contingency table                              */
/*   Left binomial probability                                                */
/*   Combinations of n things taken m at a time                               */
/*   Orderstat_tail                                                           */
/*   Quantile_conf                                                            */
/*   ROC area                                                                 */
/*   Online moments - variance, skewness, and kurtosis without storing data   */
/*                                                                            */
/******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.141592653589793

void qsortd ( int first , int last , double *data ) ;
void qsortds ( int first , int last , double *data , double *slave ) ;
void qsortdsi ( int first , int last , double *data , int *slave ) ;


/*
--------------------------------------------------------------------------------

   Normal CDF   Accurate to 7.5 e-8

--------------------------------------------------------------------------------
*/

double normal_cdf ( double z )
{
   double zz = fabs ( z ) ;
   double pdf = exp ( -0.5 * zz * zz ) / sqrt ( 2.0 * 3.141592653589793 ) ;
   double t = 1.0 / (1.0 + zz * 0.2316419) ;
   double poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t -
                     0.356563782) * t + 0.319381530) * t ;
   return (z > 0.0)  ?  1.0 - pdf * poly  :  pdf * poly ;
}

/*
--------------------------------------------------------------------------------

   Inverse normal CDF Accurate to 4.5 e-4

--------------------------------------------------------------------------------
*/

double inverse_normal_cdf ( double p )
{
   double pp = (p <= 0.5)  ?  p  :  1.0 - p ;
   double t = sqrt ( log ( 1.0 / (pp * pp) ) ) ;
   double numer = (0.010328 * t + 0.802853) * t + 2.515517 ;
   double denom = ((0.001308 * t + 0.189269) * t + 1.432788) * t + 1.0 ;
   double x = t - numer / denom ;

   return (p <= 0.5)  ?  -x  :  x ;
}

/*
--------------------------------------------------------------------------------

   Complementary error function

--------------------------------------------------------------------------------
*/

double erfc ( double x )
{
   return 2.0 - 2.0 * normal_cdf ( sqrt(2.0) * x ) ;
}

/*
--------------------------------------------------------------------------------

   Half-normal CDF == CDF of |Sn| / sqrt(n)

--------------------------------------------------------------------------------
*/

double half_normal_cdf ( double s )
{
   return 2.0 * normal_cdf ( s ) - 1.0 ;
}

/*
--------------------------------------------------------------------------------

   Gamma ( int two_K ) - Gamma distribution for twice argument
                         where twice the argument is an integer

--------------------------------------------------------------------------------
*/

double gamma_special ( int two_K ) // two_K is twice the desired argument
{
   double z = 0.5 * two_K ;

   if (two_K == 1)
      return sqrt ( PI ) ;
   else if (two_K == 2)
      return 1.0 ;
   return (z - 1.0) * gamma_special ( (int) (2.0 * z - 1.9999) ) ;
}

/*
--------------------------------------------------------------------------------

   Log Gamma (ACM algorithm 291) seems good to about 9-10 significant digits

--------------------------------------------------------------------------------
*/

double lgamma ( double x )
{
   double z, result ;

   if (x <= 0.0)
      return 0.0 ;

   if (x < 7.0) {
      result = 1.0 ;
      for (z=x ; z<7.0 ; z+=1.0) {
         result *= z ;
         x = z ;
         }
      x += 1.0 ;
      result = -log ( result ) ;
      }
   else
      result = 0.0 ;

   z = 1.0 / (x * x) ;

   return result + (x - 0.5) * log(x) - x + .918938533204673 +
      (((-.000595238095238 * z + .000793650793651) * z -
      .002777777777778) * z + .083333333333333) / x ;
}

/*
--------------------------------------------------------------------------------

   Incomplete Gamma (From Press et al)
   Note that Chi-square CDF is igamma(df/2,chisq/2)

--------------------------------------------------------------------------------
*/

double igamma ( double a , double x )
{
   int i ;
   double ap, del, sum, b, c, d, h, an ;
   double FPMIN = 1.e-30 ;

   if (x <= 0.0)
      return 0.0 ;

   if (x < (a + 1.0)) {
      ap = a ;
      del = sum = 1.0 / a ;
      for (;;) {         // Could be many iterations for some params
         ap += 1.0 ;
         del *= x / ap ;
         sum += del ;
         if (del < 1.e-8 * sum)
            break ;
         }
      return sum * exp ( a * log(x) - x - lgamma(a) ) ;
      }

   else {
      b = x + 1.0 - a ;
      c = 1.0 / FPMIN ;
      d = 1.0 / b ;
      h = d ;
      for (i=1 ; i<1000 ; i++) {
         an = i * (a - i) ;
         b += 2.0 ;
         d = an * d + b ;
         if (fabs ( d ) < FPMIN)
            d = FPMIN ;
         c = b + an / c ;
         if (fabs ( c ) < FPMIN)
            c = FPMIN ;
         d = 1.0 / d ;
         del = d * c ;
         h *= del ;
         if (fabs ( del - 1.0 ) < 1.e-8)
            break ;
         }
      return 1.0 - h * exp ( a * log(x) - x - lgamma(a) ) ;
      }
}

/*
--------------------------------------------------------------------------------

   Incomplete Beta (ACM algorithm 179 with modifications through 1976)
   Note that you can set the precision via eps and eps1.
   Higher precision can result in longer execution time.

--------------------------------------------------------------------------------
*/
double ibeta (
   double p ,   // First parameter, greater than 0
   double q ,   // First parameter, greater than 0
   double x     // Upper integration limit:  0 <= x <= 1
   )
{
   int switched_args, ib ;
   double temp, ps, px, pq, p1, d4, xb, infsum, cnt, wh ;
   double finsum, prob, term, xfac ;
   // Low precision
//   double eps = 1.e-10 ;    // Assumed machine precision
//   double eps1 = 1.e-78 ;   // Tiny representable number
   // High precision
   double eps = 1.e-12 ;    // Assumed machine precision
   double eps1 = 1.e-98 ;   // Tiny representable number
   double aleps1 = log ( eps1 ) ;

   if (x <= 0.0)       // Verify not below lower limit
      return 0.0 ;

   if (x >= 1.0)       // Or above upper limit
      return 1.0 ;

   if ((p <= 0.0)  ||  (q <= 0.0))  // If illegal parameters
      return -1.0 ;                 // Return an obvious error flag

/*
   Switch the arguments if needed for better convergence
*/

   if (x > 0.5) {
      temp = p ;
      p = q ;
      q = temp ;
      x = 1.0 - x ;
      switched_args = 1 ;
      }
   else 
      switched_args = 0 ;

/*
   Define ps as 1 if q is an integer, else q - (int) q
*/

   ps = q - (int) q ;
   if (ps == 0.0)
      ps = 1.0 ;

/*
   Compute INFSUM
*/

   px = p * log ( x ) ;
   pq = lgamma ( p + q ) ;
   p1 = lgamma ( p ) ;
   d4 = log ( p ) ;

   term = px + lgamma ( ps + p ) - lgamma ( ps ) - d4 - p1 ; // First term

   if ((int) (term / aleps1) == 0) {  // If first term does not underflow
      infsum = exp ( term ) ;
      cnt = infsum * p ;
      for (wh=1.0 ; ; wh+=1.0) {
         cnt *= (wh - ps) * x / wh ;
         term = cnt / (p + wh) ;
         infsum += term ;
         if (term / eps <= infsum)   // Avoid underflow
            break ;
         }
      }
   else 
      infsum = 0.0 ;  // First term underflowed

/*
   Compute FINSUM
*/

   finsum = 0.0 ;
   if (q <= 1.0)
      goto FINISH ;

   xb = px + q * log(1.0 - x) + pq - p1 - log(q) - lgamma ( q ) ;

   ib = (int) (xb / aleps1) ;
   if (ib < 0)
      ib = 0 ;

   xfac = 1.0 / (1.0 - x) ;
   term = exp ( xb - ib * aleps1 ) ;
   ps = q ;

   for (wh=q-1 ; wh > 0.0 ; wh-=1) {

      px = ps * xfac / (p + wh) ;

      if ((px <= 1.0)  &&  ((term / eps <= finsum)  ||  (term <= eps1 / px)))
         break ;

      ps = wh ;

      term *= px ;
      if (term > 1.0) {  // Scale if about to underflow
         --ib ;
         term *= eps1 ;
         }

      if (ib == 0)          // Only if no underflow
         finsum += term ;   // do we cumulate sum
      }

FINISH:
   prob = finsum + infsum ;

   if (switched_args)
      return 1.0 - prob ;
   else 
      return prob ;
}


/*
--------------------------------------------------------------------------------

   Inverse Incomplete Beta (Based on Jstat algorithm)

--------------------------------------------------------------------------------
*/

double inverse_ibeta (
   double p ,   // Upper integration limit:  0 <= x <= 1
   double a ,   // First parameter, greater than 0
   double b     // First parameter, greater than 0
   )
{
   int j ;
   double lna, lnb, pp, t, u, err, x, al, h, w, afac, am1, bm1 ;
   double EPS = 1e-8 ;

   if (p <= 0.0)
      return 0.0 ;

   if (p >= 1.0)
      return 1.0 ;

   am1 = a - 1 ;
   bm1 = b - 1 ;

   if (a >= 1.0  &&  b >= 1.0) {
      pp = (p < 0.5) ? p : 1.0 - p ;
      t = sqrt ( -2.0 * log(pp) ) ;
      x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t ;
      if (p < 0.5)
         x = -x ;
      al = (x * x - 3.0) / 6.0 ;
      h = 2.0 / (1.0 / (2.0 * a - 1.0)  + 1.0 / (2.0 * b - 1.0)) ;
      w = (x * sqrt(al + h) / h) - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h)) ;
      x = a / (a + b * exp(2.0 * w)) ;
      }
    
   else {
      lna = log (a / (a + b)) ;
      lnb = log (b / (a + b)) ;
      t = exp (a * lna) / a ;
      u = exp (b * lnb) / b ;
      w = t + u ;
      if (p < t / w)
         x = pow(a * w * p, 1.0 / a);
      else
         x = 1 - pow (b * w * (1.0 - p) , 1.0 / b) ;
      }

   afac = -lgamma(a) - lgamma(b) + lgamma (a + b) ;

   for (j=0 ; j<10 ; j++) {
      if (x == 0.0 || x == 1.0)
         return x ;
      err = ibeta ( a , b , x ) - p ;
      t = exp ( am1 * log(x) + bm1 * log(1.0 - x) + afac) ;
      u = err / t ;
      x -= (t = u / (1.0 - 0.5 * __min(1.0, u * (am1 / x - bm1 / (1.0 - x))))) ;
      if (x <= 0.0)  // I've never seen this true
         x = 0.5 * (x + t) ;
      if (x >= 1.0)  // I've never seen this true
         x = 0.5 * (x + t + 1.0) ;
      if (fabs(t) < EPS * x && j > 0)
         break ;
      }
   return x ;
}


/*
--------------------------------------------------------------------------------

   Student's t CDF   From Johnson et al Vol 2 Pg 364

--------------------------------------------------------------------------------
*/

double t_CDF ( int ndf , double t )
{
   double prob ;
   prob = 1.0 - 0.5 * ibeta ( 0.5 * ndf , 0.5 , ndf / (ndf + t * t) ) ;
   if (prob < 0.0)  // Trivial rounding errors can produce out-of-bound results
      prob = 0.0 ;
   if (prob > 1.0)
      prob = 1.0 ;
   if (t >= 0.0)
      return prob ;
   return 1.0 - prob ;
}


/*
--------------------------------------------------------------------------------

   Inverse Student's t CDF  based on Jstat algorithm

--------------------------------------------------------------------------------
*/

double inverse_t_CDF ( int ndf , double p )
{
   double x  ;

   x = inverse_ibeta ( 2.0 * __min ( p , 1.0 - p ) , 0.5 * ndf , 0.5 ) ;
   x = sqrt ( ndf * (1.0 - x) / x ) ;
   return (p > 0.5) ? x : -x ;
}


/*
--------------------------------------------------------------------------------

   F CDF

--------------------------------------------------------------------------------
*/

double F_CDF ( int ndf1 , int ndf2 , double F )
{
   double prob ;
   prob = 1.0 - ibeta ( 0.5 * ndf2 , 0.5 * ndf1 , ndf2 / (ndf2 + ndf1 * F) ) ;
   if (prob < 0.0)  // Trivial rounding errors can produce out-of-bound results
      prob = 0.0 ;
   if (prob > 1.0)
      prob = 1.0 ;
   return prob ;
}

/*
--------------------------------------------------------------------------------

   Poisson CDF

--------------------------------------------------------------------------------
*/

double poisson_pdf ( double lambda , int k )
{
   if (k == 0)
      return exp ( -lambda ) ;
   return exp (-lambda) * pow ( lambda , k ) / gamma_special ( 2 * k + 2 ) ;  
}

/*
--------------------------------------------------------------------------------

   AndersonDarling CDF per Marsaglia, who claims it is a good approximation

--------------------------------------------------------------------------------
*/

double AndersonDarlingCDF ( double z )
{
   if (z<.01)
      return 0.0 ;
   if (z<=2.0)
      return 2.0 * exp(-1.2337/z) * (1.0+z/8.0-0.04958*z*z/(1.325+z)) / sqrt(z) ;
   if (z<=4.0)
      return 1.0 - 0.6621361 * exp(-1.091638*z) - 0.95095 * exp(-2.005138*z) ;
   return 1.0 - 0.4938691 * exp(-1.050321*z) - 0.5946335 * exp(-1.527198*z) ;
}

/*
--------------------------------------------------------------------------------

   Kolmogorov-Smirnov CDF and its inverse.
   These are good only for fairly large n and in the tail.

--------------------------------------------------------------------------------
*/

double ks_CDF (   // Returns asymptotic CDF of KS distribution
   int n ,        // Number of cases
   double dn      // Max deviation
   )
{
   int i ;
   double term, sum, arg ;

   if ((dn <= 0.0)  ||  (n <= 0))
      return 0.0 ;

   arg = sqrt ( (double) n ) ;   // This is the von Mises fudge
   arg = arg + 0.12 + 0.11 / arg ;
   arg *= dn ;
   arg = arg * arg ;

   sum = 0.0 ;

   for (i=1 ; i<100 ; i++) { // 100 is incredibly conservative
      term = -2.0 * i * i * arg ;
      if (term < -45.0)      // This is the normal exit
         break ;             // Which usually happens very quickly
      term = exp ( term ) ;  // Avoid exp underflow problems by term check
      if (i % 2)
         sum += term ;
      else
         sum -= term ;
      }

   sum = 1.0 - 2.0 * sum ;

   if (sum < 0.0)   // This will happen for very small dn
      sum = 0.0 ;   // Which is an unavoidable property of the formula

   if (sum > 1.0)   // This should never happen
      sum = 1.0 ;   // But why take chances

   return sum ;
}

double inverse_ks ( int n , double cdf ) // Tails only, n>35
{
   return sqrt ( -log ( 0.5 * (1.0 - cdf) ) / (2.0 * n) ) ;
}

/*
--------------------------------------------------------------------------------

   Student's t test for one sample

--------------------------------------------------------------------------------
*/

double t_test ( int n , double *x )
{
   int i ;
   double mean, diff, ss, std ;

   mean = 0.0 ;
   for (i=0 ; i<n ; i++)
      mean += x[i] ;
   mean /= n ;

   ss = 0.0 ;
   for (i=0 ; i<n ; i++) {
      diff = x[i] - mean ;
      ss += diff * diff ;
      }

   std = sqrt ( ss / (n * (n - 1)) ) ;

   return mean / (std + 1.e-60) ;
}

/*
--------------------------------------------------------------------------------

   Student's t test for two samples

--------------------------------------------------------------------------------
*/

double t_test ( int n1 , double *x1 , int n2 , double *x2 )
{
   int i ;
   double mean1, mean2, ss1, ss2, diff, std ;

   mean1 = 0.0 ;
   for (i=0 ; i<n1 ; i++)
      mean1 += x1[i] ;
   mean1 /= n1 ;

   mean2 = 0.0 ;
   for (i=0 ; i<n2 ; i++)
      mean2 += x2[i] ;
   mean2 /= n2 ;

   ss1 = 0.0 ;
   for (i=0 ; i<n1 ; i++) {
      diff = x1[i] - mean1 ;
      ss1 += diff * diff ;
      }

   ss2 = 0.0 ;
   for (i=0 ; i<n2 ; i++) {
      diff = x2[i] - mean2 ;
      ss2 += diff * diff ;
      }

   std = sqrt ( (ss1 + ss2) / (n1 + n2 - 2) * (1.0 / n1 + 1.0 / n2) ) ;

   return (mean1 - mean2) / (std + 1.e-60) ;
}

/*
--------------------------------------------------------------------------------

   Mann-Whitney U-test

   This returns U based on set 1 relative to set 2, allowing a one-tailed test.
   U will be small when the mean of set 1 is greater than that of set 2.
   Note that U' = n1 * n2 - U.  Most tables perform a two-tailed test by
   considering the smaller of U and U'.
   The caller must supply int and double work vectors n1+n2 long.
   This also computes and returns the normal approximation for the one-tailed
   test, good when n1+n2 > 20.  It flips the sign of z so that z>0 when
   mean1 > mean2.

--------------------------------------------------------------------------------
*/

double U_test (
   int n1 ,
   double *x1 ,
   int n2 ,
   double *x2 ,
   int *iwork ,    // Work vector n1+n2 long
   double *work ,  // Work vector n1+n2 long
   double *z
   )
{
   int i, j, k, n, ntied ;
   double tie_correc, val, rank, dn, term1, term2, U ;


// Copy the data to one contiguous vector and sort it, preserving membership.

   for (i=0 ; i<n1 ; i++) {
      work[i] = x1[i] ;
      iwork[i] = 0 ;
      }

   for (i=0 ; i<n2 ; i++) {
      work[n1+i] = x2[i] ;
      iwork[n1+i] = 1 ;
      }

   n = n1 + n2 ;
   qsortdsi ( 0 , n-1 , work , iwork ) ;

// Compute the ranks and tie correction

   tie_correc = 0.0 ;
   for (j=0 ; j<n ; ) {
      val = work[j] ;
      for (k=j+1 ; k<n ; k++) {  // Find all ties
         if (work[k] > val)
            break ;
         }
      ntied = k - j ;
      tie_correc += (double) ntied * ntied * ntied - ntied ;
      rank = 0.5 * ((double) j + (double) k + 1.0) ;
      while (j < k)
         work[j++] = rank ;
      } // For each case in sorted x array

   // Compute the U statistic

   U = 0.0 ;
   for (i=0 ; i<n ; i++) {
      if (! iwork[i])
         U += work[i] ;
      }

   U = n1 * n2 + 0.5 * (n1 * (n1 + 1.0)) - U ;

   // Compute the normal approximation

   dn = n ;
   term1 = n1 * n2 / (dn * (dn - 1.0)) ;
   term2 = (dn * dn * dn - dn - tie_correc) / 12.0 ;
   *z = (0.5 * n1 * n2 - U) / sqrt ( term1 * term2 ) ;
 
   return U ;
}

/*
--------------------------------------------------------------------------------

   Kolmogorov-Smirnov test for x (sorted) following a uniform distribution

--------------------------------------------------------------------------------
*/

double ks_test ( int n , double *x , double *D_plus , double *D_minus )
{
   int i ;
   double fn, old_fn ;

   *D_plus = *D_minus = old_fn = 0.0 ;

   for (i=0 ; i<n ; i++) {
      fn = (i + 1.0) / n ;
      if (fn - x[i] > *D_plus)
         *D_plus = fn - x[i] ;
      if (x[i] - old_fn > *D_minus)
         *D_minus = x[i] - old_fn ;
      old_fn = fn ;
      }

   if (*D_plus >= *D_minus)
      return *D_plus ;
   return *D_minus ;
}

/*
--------------------------------------------------------------------------------

   Anderson-Darling test for x (sorted) following a uniform distribution

--------------------------------------------------------------------------------
*/

double AndersonDarlingTest ( int n , double *x )
{
   int i ;
   double term, z ;

   qsortd ( 0 , n-1 , x ) ;

   z = -1.0 * n * n ;
   for (i=0 ; i<n ; ++i) {
      term = x[i] * (1.0 - x[n-i-1]) ;
      if (term < 1.e-30)  // This pathological condition damages the test
         term = 1.e-30 ;
      z -= (2.0 * i + 1.0) * log ( term ) ;
      }

   term = AndersonDarlingCDF ( z / n ) ;
   return 1.0 - term ;
}

/*
--------------------------------------------------------------------------------

   One-way ANOVA

--------------------------------------------------------------------------------
*/

double ANOVA_1 (    // Returns F ratio
   int n ,          // Total number of cases
   int K ,          // Number of groups
   double *x ,      // The measurements are here
   int *id ,        // Group ID of each case, 0 through k-1
   double *account ,// Returns between / (between + within)
   double *pval ,   // Returns right-tail p-value
   int *counts ,    // Work vector K long
   double *means    // Work vector K long
   )
{
   int i, k ;
   double grand_mean, diff, between, within ;

   // Compute the grand and group means

   grand_mean = 0.0 ;
   for (k=0 ; k<K ; k++) {
      counts[k] = 0 ;
      means[k] = 0.0 ;
      }

   for (i=0 ; i<n ; i++) {
      k = id[i] ;
      if (k < 0  ||  k >= K)  // A careful user will never let this happen
         return -1.e60 ;      // Get somebody's attention!
      ++counts[k] ;
      grand_mean += x[i] ;
      means[k] += x[i] ;
      }

   grand_mean /= n ;
   for (k=0 ; k<K ; k++)
      means[k] /= counts[k] + 1.e-60 ;

   // Compute SumSquares between

   between = 0.0 ;
   for (k=0 ; k<K ; k++) {
      diff = means[k] - grand_mean ;
      between += counts[k] * diff * diff ;
      }
   between /= K - 1 ;

   // Compute SumSquares within

   within = 0.0 ;
   for (i=0 ; i<n ; i++) {
      diff = x[i] - means[id[i]] ;
      within += diff * diff ;
      }
   within /= n - K ;

   *account = between / (between + within) ;
   *pval = 1.0 - F_CDF ( K-1 , n-K , between / (within + 1.e-60) ) ;

   return between / (within + 1.e-60) ;
}

/*
--------------------------------------------------------------------------------

   Kruskal-Wallis nonparametric ANOVA

--------------------------------------------------------------------------------
*/

double kruskal_wallis (
   int n ,          // Total number of cases
   int K ,          // Number of groups
   double *x ,      // The measurements are here
   int *id ,        // Group ID of each case, 0 through k-1
   double *work ,   // Work vector n long
   int *iwork       // Work vector n long
   )
{
   int i, j, k, nn, ntied ;
   double val, sum, tie_correc, rank, kw ;

   memcpy ( work , x , n * sizeof(double) ) ;
   memcpy ( iwork , id , n * sizeof(int) ) ;
   qsortdsi ( 0 , n-1 , work , iwork ) ;

// Compute the ranks and tie correction

   tie_correc = 0.0 ;
   for (j=0 ; j<n ; ) {
      val = work[j] ;
      for (k=j+1 ; k<n ; k++) {  // Find all ties
         if (work[k] > val)
            break ;
         }
      ntied = k - j ;
      tie_correc += (double) ntied * ntied * ntied - ntied ;
      rank = 0.5 * ((double) j + (double) k + 1.0) ;
      while (j < k)
         work[j++] = rank ;
      } // For each case in sorted x array

   // Compute the Kruskal-Wallis statistic

   kw = 0.0 ;
   for (k=0 ; k<K ; k++) {    // For all classes
      nn = 0 ;
      sum = 0.0 ;
      for (i=0 ; i<n ; i++) { // Sum the ranks of class k, and count membership
         if (iwork[i] == k) {
            ++nn ;
            sum += i ;
            }
         }
      kw += (sum + nn) * (sum + nn) / (nn + 1.e-60) ; // k is org 0, ranks org 1
      }

   kw = 12.0 / (n * (n+1)) * kw - 3.0 * (n + 1) ;
   kw /= 1.0 - tie_correc / ((double) n * n * n - n) ;

   return kw ;
}


/*
--------------------------------------------------------------------------------

   Chi-square test

--------------------------------------------------------------------------------
*/

void chisq (
   int nrows,       // Number of rows in rectangular data matrix
   int ncols,       // And columns
   int *data,       // Nrows by ncols (changes fastest) matrix of cell counts
   double *csq,     // Returns chi-square
   double *contin,  // Returns contingency coefficient
   double *CramerV, // Returns Cramer's V
   double *pval,    // Returns right-tail p-value
   int *rmarg,      // Work vector nrows long
   int *cmarg       // Work vector ncols long
   )
{
   int irow, icol, total, ndf ;
   double diff, expected, chi_square ;

   ndf = (nrows-1) * (ncols-1) ;

   if (ndf == 0) {   // Careless user!
      *csq = 0.0 ;
      *contin = 0.0 ;
      *CramerV = 0.0 ;
      *pval = 1.0 ;
      return ;
      }

   total = 0 ;

   for (irow=0 ; irow<nrows ; irow++) {
      rmarg[irow] = 0 ;
      for (icol=0 ; icol<ncols ; icol++)
         rmarg[irow] += data[irow*ncols+icol] ;
      total += rmarg[irow] ;
      }

   for (icol=0 ; icol<ncols ; icol++) {
      cmarg[icol] = 0 ;
      for (irow=0 ; irow<nrows ; irow++)
         cmarg[icol] += data[irow*ncols+icol] ;
      }

   chi_square = 0.0 ;
   for (irow=0 ; irow<nrows ; irow++) {
      for (icol=0 ; icol<ncols ; icol++) {
         expected = (double) rmarg[irow] * (double) cmarg[icol] / (total + 1.e-20) ;
         diff = data[irow*ncols+icol] - expected ;
         chi_square += diff * diff / (expected + 1.e-20) ;
         }
      }

   *csq = chi_square ;
   *contin = sqrt ( chi_square / (total + chi_square) ) ;
   *CramerV = chi_square / total ;
   if (nrows < ncols)
      *CramerV /= nrows - 1 ;
   else
      *CramerV /= ncols - 1 ;
   *CramerV = sqrt ( *CramerV ) ;
   *pval = 1.0 - igamma ( 0.5 * ndf , 0.5 * chi_square ) ;
}   


/*
--------------------------------------------------------------------------------

   nominal_lambda - Compute Lambda for a pair of nominal variables

--------------------------------------------------------------------------------
*/

void nominal_lambda (
   int nrows,       // Number of rows in data
   int ncols,       // And columns
   int *data,       // Nrows by ncols (changes fastest) matrix of cell counts
   double *row_dep, // Returns asymmetric lambda when row is dependent
   double *col_dep, // Returns asymmetric lambda when column is dependent
   double *sym      // Returns symmetric lambda
   )
{
   int irow, icol, total, row_cell_max, col_cell_max, sum_row_cell_max, sum_col_cell_max ;
   int row_total, col_total, max_row_total, max_col_total, numer, denom ;

   if (nrows < 2  ||  ncols < 2) {   // Careless user!
      *row_dep = *col_dep = *sym = 0.0 ;
      return ;
      }

   total = 0 ;

   sum_row_cell_max = 0 ;
   max_row_total = 0 ;
   for (irow=0 ; irow<nrows ; irow++) {
      row_cell_max = 0 ;
      row_total = 0 ;
      for (icol=0 ; icol<ncols ; icol++) {
         if (data[irow*ncols+icol] > row_cell_max)
            row_cell_max = data[irow*ncols+icol] ;
         row_total += data[irow*ncols+icol] ;
         total += data[irow*ncols+icol] ;
         }
      if (row_total > max_row_total)
         max_row_total = row_total ;
      sum_row_cell_max += row_cell_max ;
      }

   sum_col_cell_max = 0 ;
   max_col_total = 0 ;
   for (icol=0 ; icol<ncols ; icol++) {
      col_cell_max = 0 ;
      col_total = 0 ;
      for (irow=0 ; irow<nrows ; irow++) {
         if (data[irow*ncols+icol] > col_cell_max)
            col_cell_max = data[irow*ncols+icol] ;
         col_total += data[irow*ncols+icol] ;
         }
      if (col_total > max_col_total)
         max_col_total = col_total ;
      sum_col_cell_max += col_cell_max ;
      }

   if (total > max_row_total)
      *row_dep = (double) (sum_col_cell_max - max_row_total) / (double) (total - max_row_total) ;
   else
      *row_dep = 1.0 ;
   if (total > max_col_total)
      *col_dep = (double) (sum_row_cell_max - max_col_total) / (double) (total - max_col_total) ;
   else
      *col_dep = 1.0 ;
   numer = sum_col_cell_max - max_row_total + sum_row_cell_max - max_col_total ;
   denom = 2 * total - max_row_total - max_col_total ;
   if (denom > 0)
      *sym = (double) numer / (double) denom ;
   else
      *sym = 1.0 ;
}   


/*
--------------------------------------------------------------------------------

   uncert_reduc - Uncertainty reduction for a pair of nominal variables

--------------------------------------------------------------------------------
*/

void uncert_reduc (
   int nrows,       // Number of rows in data
   int ncols,       // And columns
   int *data,       // Nrows by ncols (changes fastest) matrix of cell counts
   double *row_dep, // Returns asymmetric UR when row is dependent
   double *col_dep, // Returns asymmetric UR when column is dependent
   double *sym,     // Returns symmetric UR
   int *rmarg,      // Work vector nrows long
   int *cmarg       // Work vector ncols long
   )
{
   int irow, icol, total ;
   double p, numer, Urow, Ucol, Ujoint ;

   if (nrows < 2  ||  ncols < 2) {   // Careless user!
      *row_dep = *col_dep = *sym = 0.0 ;
      return ;
      }

   total = 0 ;

   for (irow=0 ; irow<nrows ; irow++) {
      rmarg[irow] = 0 ;
      for (icol=0 ; icol<ncols ; icol++)
         rmarg[irow] += data[irow*ncols+icol] ;
      total += rmarg[irow] ;
      }

   for (icol=0 ; icol<ncols ; icol++) {
      cmarg[icol] = 0 ;
      for (irow=0 ; irow<nrows ; irow++)
         cmarg[icol] += data[irow*ncols+icol] ;
      }

   Urow = 0.0 ;
   for (irow=0 ; irow<nrows ; irow++) {
      if (rmarg[irow]) {
         p = (double) rmarg[irow] / (double) total ;
         Urow -= p * log ( p ) ;
         }
      }

   Ucol = 0.0 ;
   for (icol=0 ; icol<ncols ; icol++) {
      if (cmarg[icol]) {
         p = (double) cmarg[icol] / (double) total ;
         Ucol -= p * log ( p ) ;
         }
      }

   Ujoint = 0.0 ;
   for (irow=0 ; irow<nrows ; irow++) {
      for (icol=0 ; icol<ncols ; icol++) {
         if (data[irow*ncols+icol]) {
            p = (double) data[irow*ncols+icol] / (double) total ;
            Ujoint -= p * log ( p ) ;
            }
         }
      }

   numer = Urow + Ucol - Ujoint ;
   if (Urow > 0)
      *row_dep = numer / Urow ;
   else
      *row_dep = 0.0 ;
   if (Ucol > 0)
      *col_dep = numer / Ucol ;
   else
      *col_dep = 0.0 ;
   if (Urow + Ucol > 0)
      *sym = 2.0 * numer / (Urow + Ucol) ;
   else
      *sym = 0.0 ;
}   


/*
--------------------------------------------------------------------------------

   Evaluate the probability that a binomial (n,p) will be
   less than or equal to m.

--------------------------------------------------------------------------------
*/

double left_binomial (
   int n ,
   double p ,
   int m
   )
{
   if (m >= n)
      return 1.0 ;
   if (m < 0)
      return 0.0 ;
   return 1.0 - ibeta ( m+1 , n-m , p ) ;
}

/*
--------------------------------------------------------------------------------

   Combinations of n things taken m at a time
   This is exact, but good only for fairly small n.

--------------------------------------------------------------------------------
*/

double combinations ( int n , int m )
{
   int j ;
   double product=1.0 ;

   if (m < n-m)
      j = m ;
   else
      j = n-m ;
   while (j)
      product *= (double) n-- / (double) j-- ;
   return product ;
}

/*
--------------------------------------------------------------------------------

   Evaluate the probability of the m'th order statistic exceeding
   the q'th quantile.

--------------------------------------------------------------------------------
*/

double orderstat_tail (
   int n ,
   double q ,
   int m
   )
{
   if (m > n)
      return 1.0 ;
   if (m <= 0)
      return 0.0 ;
//   if (n >= 100000) {
//      double std = sqrt ( n * q * (1.0 - q) ) ;
//      return normal_cdf ( (m - n * q) / std ) ;
//      }
   return 1.0 - ibeta ( m , n-m+1 , q ) ;
}

/*
--------------------------------------------------------------------------------

   Compute the p such that probability of the m'th order statistic
   exceeding the p'th quantile equals the specified conf.
   Generally, set conf=.05 or so.

   This solves the equation conf-orderstat_tail=0.
   First, the root is bounded by locating where the function changes sign.
   Then Ridder's method is used to rapidly refine the root.

--------------------------------------------------------------------------------
*/

#define QCEPS 1.e-10

double quantile_conf (  // Returns pessimistic value of quantile
   int n ,       // Number of cases
   int m ,       // Order stat used; 1+number of discarded tail cases
   double conf   // Desired confidence level for quantile, often 0.05 or so
   )
{
   int iter ;
   double x1, y1, x2, y2, x3, y3, x, y, denom ;

   x1 = 0.0 ;         // Leftmost extreme has p=0,
   y1 = conf - 1.0 ;  // At which left_binom=1.

   x2 = y2 = x3 = y3 = 0.0 ;  // Not needed.  Shuts up LINT.

/*
   Advance until the sign changes.  This should be very soon.
*/

   for (x3=0.1 ; x3<=1.0 ; x3+=0.0999999999) {
      y3 = conf - orderstat_tail ( n , x3 , m ) ;  // Function value here
      if (fabs ( y3 ) < QCEPS)  // Only by wildest good luck
         return x3 ;            // But should check anyway
      if (y3 > 0.0)             // Did we just change sign?
         break ;                // If so, root is bracketed
      x1 = x3 ;                 // Not yet bracketed, so advance
      y1 = y3 ;
      }

/*
   The root is now bracketed.  Refine using Ridder's method.
*/

   for (iter=0 ; iter<200 ; iter++) {  // In practice, iter=6 or so max!

      x2 = 0.5 * (x1 + x3) ;           // Midpoint of bracketing interval
      if (x3 - x1 < QCEPS)             // If the interval is tiny
         return x2 ;                   // Quit right now

      y2 = conf - orderstat_tail ( n , x2 , m ) ;  // Function value here
      if (fabs ( y2 ) < QCEPS)  // If the root is real close
         return x2 ;            // We are done

      denom = sqrt ( y2 * y2 - y1 * y3 ) ;      // Use Ridder's method
      x = x2 + (x1 - x2) * y2 / denom ;         // To find a new trial point
      y = conf - orderstat_tail ( n , x , m ) ;  // Function value here
      if (fabs ( y ) < QCEPS)                   // If the root is real close
         return x ;                             // We are done

      if ((y2 < 0.0)  &&  (y > 0.0)) { // Root between midpoint and trial x?
         x1 = x2 ;
         y1 = y2 ;
         x3 = x ;
         y3 = y ;
         }
      else if ((y < 0.0)  &&  (y2 > 0.0)) { // Ditto, but other way?
         x1 = x ;
         y1 = y ;
         x3 = x2 ;
         y3 = y2 ;
         }
      else if (y < 0.0) {  // Must keep one of the current bounds
         x1 = x ;
         y1 = y ;
         }
      else {               // Ditto, but other bound
         x3 = x ;
         y3 = y ;
         }
      } // For iters

   return x2 ;   // Should never happen because convergence is fast
}

/*
--------------------------------------------------------------------------------

   Compute area under the ROC curve, centering target if center != 0

   WARNING: This changes pred and target!

--------------------------------------------------------------------------------
*/

double ROCarea ( int n , double *pred , double *target , int center )
{
   int i ;
   double mean, win_sum, lose_sum, win, lose, roc ;

   if (center) {
      mean = 0.0 ;
      for (i=0 ; i<n ; i++)
         mean += target[i] ;
      mean /= n ;
      for (i=0 ; i<n ; i++)
         target[i] -= mean ;
      }

   qsortds ( 0 , n-1 , pred , target ) ;

   win_sum = lose_sum = win = lose = roc = 0.0 ;

   for (i=0 ; i<n ; i++) {
      if (target[i] > 0.0)
         win_sum += target[i] ;
      else
         lose_sum -= target[i] ;
      }

   if (win_sum == 0.0  ||  lose_sum == 0.0) // Degenerate case
      return 0.5 ;

   for (i=n-1 ; i>=0 ; i--) {
      if (target[i] > 0.0)
         win += target[i] / win_sum ;
      else if (target[i] < 0.0)
         roc -= win * target[i] / lose_sum ;
      }

   return roc ;
}


/*
--------------------------------------------------------------------------------

   Compute the running mean, variance, skewness, and kurtosis without having
   to store the data.  Numerically stable, even under large offset from zero.
   It can handle multiple data streams simultaneously.
   The work vectoprs must be as long as there are streams.
   So if only one stream is used, the works can be scalars.

--------------------------------------------------------------------------------
*/

static int nstreams, n ;
static double *delta, *mean, *sum2, *sum3, *sum4 ;

void online_reset ( int ns , double *work1 , double *work2 , double *work3 , double *work4 , double *work5 )
{
   nstreams = ns ;
   delta = work1 ;
   mean = work2 ;
   sum2 = work3 ;
   sum3 = work4 ;
   sum4 = work5 ;
   n = 0 ;
}

void online_update ( double *y )
{
   int i ;
   double dsquare, np1 ;

   if (n == 0) {
      for (i=0 ; i<nstreams ; i++) {
         mean[i] = y[i] ;
         sum2[i] = sum3[i] = sum4[i] = 0.0 ;
         }
      n = 1 ;
      return ;
      }

   np1 = n + 1.0 ;

   for (i=0 ; i<nstreams ; i++) {
      delta[i] = y[i] - mean[i] ;
      mean[i] += delta[i] / np1 ;
      dsquare = delta[i] * delta[i] ;
      sum4[i] += n * ((double) n * n - n + 1.0) * dsquare * dsquare / (np1 * np1 * np1) ;
      sum4[i] += 6.0 * sum2[i] * dsquare / (np1 * np1) ;
      sum4[i] -= 4.0 * sum3[i] * delta[i] / np1 ;
      sum3[i] += n * (n - 1.0) * dsquare * delta[i] / (np1 * np1) ;
      sum3[i] -= 3.0 * sum2[i] * delta[i] / np1 ;
      sum2[i] += n * dsquare / np1 ;
      }
   ++n ;
}


void online_get_mean ( double *value )
{
   int i ;

   for (i=0 ; i<nstreams ; i++)
      value[i] = mean[i] ;
}


void online_get_variance ( double *value )
{
   int i ;

   for (i=0 ; i<nstreams ; i++)
      value[i] = sum2[i] / n ;
}


void online_get_skewness ( double *value )
{
   int i ;
   double std ;

   for (i=0 ; i<nstreams ; i++) {
      std = sqrt ( sum2[i] / n ) ;
      value[i] = sum3[i] / (n * std * std * std) ;
      }
}

void online_get_kurtosis ( double *value )
{
   int i ;
   double var ;

   for (i=0 ; i<nstreams ; i++) {
      var = sum2[i] / n ;
      value[i] = sum4[i] / (n * var * var) ;
      }
}
