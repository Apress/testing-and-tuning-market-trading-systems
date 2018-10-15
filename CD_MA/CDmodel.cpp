// CDmodel - Coordinate Descent class

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#define RESULTS 0

class CoordinateDescent {

friend double cv_train ( int n , int nvars , int nfolds , double *xx , double *yy , double *ww ,
                       double *lambdas , double *lambda_OOS , double *work , int covar_updates ,
                       int n_lambda , double alpha , int maxits , double eps , int fast_test ) ;

public:

   CoordinateDescent ( int nvars , int ncases , int weighted , int covar_updates , int n_lambda ) ;
   ~CoordinateDescent () ;
   void get_data ( int istart , int n , double *x , double *y , double *w ) ;
   void core_train ( double alpha , double lambda , int maxits , double eps , int fast_test , int warm_start ) ;
   double get_lambda_thresh ( double alpha ) ;
   void lambda_train ( double alpha , int maxits , double eps , int fast_test , double max_lambda , int print_steps ) ;

   int ok ;              // Was everything legal and allocs successful?
   double *beta ;        // Beta coefs (nvars of them)
   double explained ;    // Fraction of variance explained by model; computed by core_train()
   double *Xmeans ;      // Mean of each X predictor
   double *Xscales ;     // And standard deviation
   double Ymean ;        // Intercept (mean of Y)
   double Yscale ;       // Standard deviation of Y


private:
   int nvars ;           // Number of variables
   int ncases ;          // Number of cases
   int covar_updates ;   // Does user want (often faster) covariance update method?
   int n_lambda ;        // Reserve space for this many beta sets for saving by lambda_train() (may be zero)
   double *lambda_beta ; // Saved beta coefs (n_lambda sets of nvars of them)
   double *lambdas ;     // Lambdas tested by lambda_train()
   double *x ;           // Normalized (mean=0, std=1) X; ncases by nvars
   double *y ;           // Normalized (mean=0, std=1) Y
   double *w ;           // Weight of each case, or NULL if equal weighting
   double *resid ;       // Residual
   double *Xinner ;      // Nvars square inner product matrix if covar_updates
   double *Yinner ;      // Nvars XY inner product vector if covar_updates
   double *XSSvec ;      // If cases are weighted, this is weighted X sumsquares
} ;

/*
-----------------------------------------------------------------

   Constructor and destructor

-----------------------------------------------------------------
*/

CoordinateDescent::CoordinateDescent (
   int nv ,   // Number of predictor variables
   int nc ,   // Number of cases we will be training
   int wtd ,  // Will we be using case weights?
   int cu ,   // Use fast covariance updates rather than slow naive method
   int nl     // Number of lambdas we will be using in training
   )
{
   nvars = nv ;
   ncases = nc ;
   covar_updates = cu ;
   n_lambda = nl ;

   ok = 1 ;     //Be optimistic

   x = (double *) malloc ( ncases * nvars * sizeof(double) ) ;
   y = (double *) malloc ( ncases * sizeof(double) ) ;
   Xmeans = (double *) malloc ( nvars * sizeof(double) ) ;
   Xscales = (double *) malloc ( nvars * sizeof(double) ) ;
   beta = (double *) malloc ( nvars * sizeof(double) ) ;
   resid = (double *) malloc ( ncases * sizeof(double) ) ;

   if (wtd) {  // Will be using weighted cases?
      w = (double *) malloc ( ncases * sizeof(double) ) ;
      XSSvec = (double *) malloc ( nvars * sizeof(double) ) ;
      }
   else
      w = XSSvec = NULL ;

   if (covar_updates) {  // Alloc Xinner full square for faster addressing, despite symmetry
      Xinner = (double *) malloc ( nvars * nvars * sizeof(double) ) ;
      Yinner = (double *) malloc ( nvars * sizeof(double) ) ;
      }
   else
      Xinner = Yinner = NULL ;

   if (n_lambda > 0) {
      lambda_beta = (double *) malloc ( n_lambda * nvars * sizeof(double) ) ;
      lambdas = (double *) malloc ( n_lambda * sizeof(double) ) ;
      }
   else
      lambda_beta = lambdas = NULL ;

   if (x == NULL  ||  y == NULL  ||  Xmeans == NULL  ||  Xscales == NULL  ||
       beta == NULL  ||  resid == NULL  ||  (wtd && w == NULL)  ||  (wtd && XSSvec == NULL)  ||
       (covar_updates && Xinner == NULL)  ||  (covar_updates && Yinner == NULL)  ||
       (n_lambda > 0 && lambda_beta == NULL)  ||  (n_lambda > 0 && lambdas == NULL)) {
      if (x != NULL) {
         free ( x ) ;
         x = NULL ;
         }
      if (y != NULL) {
         free ( y ) ;
         y = NULL ;
         }
      if (w != NULL) {
         free ( w ) ;
         w = NULL ;
         }
      if (XSSvec != NULL) {
         free ( XSSvec ) ;
         XSSvec = NULL ;
         }
      if (Xmeans != NULL) {
         free ( Xmeans ) ;
         Xmeans = NULL ;
         }
      if (Xscales != NULL) {
         free ( Xscales ) ;
         Xscales = NULL ;
         }
      if (beta != NULL) {
         free ( beta ) ;
         beta = NULL ;
         }
      if (resid != NULL) {
         free ( resid ) ;
         resid = NULL ;
         }
      if (Xinner != NULL) {
         free ( Xinner ) ;
         Xinner = NULL ;
         }
      if (Yinner != NULL) {
         free ( Yinner ) ;
         Yinner = NULL ;
         }
      if (lambda_beta != NULL) {
         free ( lambda_beta ) ;
         lambda_beta = NULL ;
         }
      if (lambdas != NULL) {
         free ( lambdas ) ;
         lambdas = NULL ;
         }
      ok = 0 ;
      return ;
      }
}


CoordinateDescent::~CoordinateDescent ()
{
   if (x != NULL)
      free ( x ) ;
   if (y != NULL)
      free ( y ) ;
   if (w != NULL)
      free ( w ) ;
   if (XSSvec != NULL)
      free ( XSSvec ) ;
   if (Xmeans != NULL)
      free ( Xmeans ) ;
   if (Xscales != NULL)
      free ( Xscales ) ;
   if (beta != NULL)
      free ( beta ) ;
   if (resid != NULL)
      free ( resid ) ;
   if (Xinner != NULL)
      free ( Xinner ) ;
   if (Yinner != NULL)
      free ( Yinner ) ;
   if (lambda_beta != NULL)
      free ( lambda_beta ) ;
   if (lambdas != NULL)
      free ( lambdas ) ;
}


/*
-----------------------------------------------------------------

   Get and standardize the data
   Also compute inner products if covar_update

-----------------------------------------------------------------
*/

void CoordinateDescent::get_data (
   int istart ,   // Starting index in full database for getting ncases of training set
   int n ,        // Number of cases in full database (we wrap back to the start if needed)
   double *xx ,   // Full database (n rows, nvars columns)
   double *yy ,   // Predicted variable vector, n long
   double *ww     // Case weights (n long) or NULL if no weighting
   )
{
   int icase, ivar, jvar, k ;
   double sum, xm, xs, diff, *xptr ;

/*
   Standardize X
*/

   for (ivar=0 ; ivar<nvars ; ivar++) {

      xm = 0.0 ;
      for (icase=0 ; icase<ncases ; icase++) {
         k = (icase + istart) % n ;
         xm += xx[k*nvars+ivar] ;
         }
      xm /= ncases ;
      Xmeans[ivar] = xm ;

      xs = 1.e-60 ;  // Prevent division by zero later
      for (icase=0 ; icase<ncases ; icase++) {
         k = (icase + istart) % n ;
         diff = xx[k*nvars+ivar] - xm ;
         xs += diff * diff ;
         }
      xs = sqrt ( xs / ncases ) ;
      Xscales[ivar] = xs ;

      for (icase=0 ; icase<ncases ; icase++) {
         k = (icase + istart) % n ;
         x[icase*nvars+ivar] = (xx[k*nvars+ivar] - xm) / xs ;
         }
      }

/*
   Standardize Y
*/

   Ymean = 0.0 ;
   for (icase=0 ; icase<ncases ; icase++) {
      k = (icase + istart) % n ;
      Ymean += yy[k] ;
      }
   Ymean /= ncases ;

   Yscale = 1.e-60 ;  // Prevent division by zero later
   for (icase=0 ; icase<ncases ; icase++) {
      k = (icase + istart) % n ;
      diff = yy[k] - Ymean ;
      Yscale += diff * diff ;
      }
   Yscale = sqrt ( Yscale / ncases ) ;

   for (icase=0 ; icase<ncases ; icase++) {
      k = (icase + istart) % n ;
      y[icase] = (yy[k] - Ymean) / Yscale ;
      }

/*
   If weighted, scale weights to sum to 1.0
   and then compute the weighted X sum of squares
*/

   if (w != NULL) {
      assert ( ww != NULL) ;
      sum = 0.0 ;
      for (icase=0 ; icase<ncases ; icase++) {
         k = (icase + istart) % n ;
         w[icase] = ww[k] ;
         sum += w[icase] ;
         }
      for (icase=0 ; icase<ncases ; icase++)
         w[icase] /= sum ;

      for (ivar=0 ; ivar<nvars ; ivar++) {
         xptr = x + ivar ;
         sum = 0.0 ;
         for (icase=0 ; icase<ncases ; icase++)
            sum += w[icase] * xptr[icase*nvars] * xptr[icase*nvars] ;
         XSSvec[ivar] = sum ;
         }
      }

/*
   If user requests covariance updates, compute required inner products
   We store the full Xinner matrix for faster addressing later,
   even though it is symmetric.
   We handle both unweighted and weighted cases here.
*/

   if (covar_updates) {
      for (ivar=0 ; ivar<nvars ; ivar++) {
         xptr = x + ivar ;

         // Do XiY
         sum = 0.0 ;
         if (w != NULL) {  // Weighted cases
            for (icase=0 ; icase<ncases ; icase++)
               sum += w[icase] * xptr[icase*nvars] * y[icase] ;
            Yinner[ivar] = sum ;
            }
         else {
            for (icase=0 ; icase<ncases ; icase++)
               sum += xptr[icase*nvars] * y[icase] ;
            Yinner[ivar] = sum / ncases ;
            }

         // Do XiXj
         if (w != NULL) {  // Weighted
            for (jvar=0 ; jvar<nvars ; jvar++) {
               if (jvar == ivar)
                  Xinner[ivar*nvars+jvar] = XSSvec[ivar] ; // Already computed, so might as well use it
               else if (jvar < ivar)                       // Matrix is symmetric, so just copy
                  Xinner[ivar*nvars+jvar] = Xinner[jvar*nvars+ivar] ;
               else {
                  sum = 0.0 ;
                  for (icase=0 ; icase<ncases ; icase++)
                     sum += w[icase] * xptr[icase*nvars] * x[icase*nvars+jvar] ;
                  Xinner[ivar*nvars+jvar] = sum ;
                  }               
               }
            } // If w

         else {  // Unweighted
            for (jvar=0 ; jvar<nvars ; jvar++) {
               if (jvar == ivar)
                  Xinner[ivar*nvars+jvar] = 1.0 ;      // Recall that X is standardized
               else if (jvar < ivar)                   // Matrix is symmetric, so just copy
                  Xinner[ivar*nvars+jvar] = Xinner[jvar*nvars+ivar] ;
               else {
                  sum = 0.0 ;
                  for (icase=0 ; icase<ncases ; icase++)
                     sum += xptr[icase*nvars] * x[icase*nvars+jvar] ;
                  Xinner[ivar*nvars+jvar] = sum / ncases ;
                  }
               }
            } // Else not weighted
         } // For ivar
      }
}



/*
-----------------------------------------------------------------

   Core training routine

   Note that fast versus slow test can sometimes significantly
   change the model!  This is because when convergence of an
   active-set-only pass is flagged, we do a complete pass.
   This can suddenly change the active set, with divergence
   thereafter.  I have seen this only for ncases<nvars,
   an inherently unstable application.

-----------------------------------------------------------------
*/

void CoordinateDescent::core_train (
   double alpha ,    // User-specified alpha, (0,1) (0 problematic for descending lambda)
   double lambda ,   // Can be user-specified, but usually from lambda_train()
   int maxits ,      // Maximum iterations, for safety only
   double eps ,      // Convergence criterion, typically 1.e-5 or so
   int fast_test ,   // Base convergence on max beta change vs explained variance?
   int warm_start    // Start from existing beta, rather than zero?
   )
{
   int i, iter, icase, ivar, kvar, do_active_only, active_set_changed, converged ;
   double *xptr, residual_sum, S_threshold, argument, new_beta, correction, update_factor ;
   double sum, explained_variance, crit, prior_crit, penalty, max_change, Xss, YmeanSquare ;


/*
   Initialize
*/

   S_threshold = alpha * lambda ;   // Threshold for the soft-thresholding operator S()
   do_active_only = 0 ;             // Begin with a complete pass
   prior_crit = 1.0e60 ;            // For convergence test

   if (warm_start) {                // Pick up with current betas?
      if (! covar_updates) {        // If not using covariance updates, must recompute residuals
         for (icase=0 ; icase<ncases ; icase++) {
            xptr = x + icase * nvars ;
            sum = 0.0 ;
            for (ivar=0 ; ivar<nvars ; ivar++)
               sum += beta[ivar] * xptr[ivar] ;
            resid[icase] = y[icase] - sum ;
            }
         }
      }

   else {                           // Not warm start, so initial betas are all zero
      for (i=0 ; i<nvars ; i++)
         beta[i] = 0.0 ;
      for (i=0 ; i<ncases ; i++)    // Initial residuals are just the Y variable
         resid[i] = y[i] ;
      }

   // YmeanSquare will remain fixed throughout training.
   // Its only use is for computing explained variance for the user's edification.

   if (w != NULL) {                  // We need weighted squares to evaluate explained variance
      YmeanSquare = 0.0 ;
      for (i=0 ; i<ncases ; i++)
         YmeanSquare += w[i] * y[i] * y[i] ;
      }
   else
      YmeanSquare = 1.0 ;


/*
   Outmost loop iterates until converged or user's maxits limit hit

*/

   for (iter=0 ; iter<maxits ; iter++) {

/*
   Pass through variables
*/

      active_set_changed = 0 ;  // Did any betas go to/from 0.0?
      max_change = 0.0 ;        // For fast convergence test

      for (ivar=0 ; ivar<nvars ; ivar++) {  // Descend on this beta

         if (do_active_only  &&  beta[ivar] == 0.0)
            continue ;

         // Denominator in update
         if (w != NULL)      // Weighted?
            Xss = XSSvec[ivar] ;
         else
            Xss = 1 ;        // X was standardized
         update_factor = Xss + lambda * (1.0 - alpha) ;

         if (covar_updates) {   // Any sensible user will specify this unless ncases < nvars
            sum = 0.0 ;
            for (kvar=0 ; kvar<nvars ; kvar++)
               sum += Xinner[ivar*nvars+kvar] * beta[kvar] ;
            residual_sum = Yinner[ivar] - sum ;
            argument = residual_sum + Xss * beta[ivar] ;   // Argument to S() [MY FORMULA]
            }

         else if (w != NULL) {           // Use slow definitional formula (okay if ncases < nvars)
            argument = 0.0 ;
            xptr = x + ivar ;    // Point to column of this variable
            for (icase=0 ; icase<ncases ; icase++)     // Argument to S()  [Their Eq 10)
               argument += w[icase] * xptr[icase*nvars] * (resid[icase] + beta[ivar] * xptr[icase*nvars])  ;
            }

         else {                          // Use slow definitional formula (okay if ncases < nvars)
            residual_sum = 0.0 ;
            xptr = x + ivar ;    // Point to column of this variable
            for (icase=0 ; icase<ncases ; icase++)
               residual_sum += xptr[icase*nvars] * resid[icase] ;  // X_ij * RESID_i
            residual_sum /= ncases ;
            argument = residual_sum + beta[ivar] ;  // Argument to S() ;    (Eq 8)
            }

         // Apply the soft-thresholding operator S()

         if (argument > 0.0  &&  S_threshold < argument)
            new_beta = (argument - S_threshold) / update_factor ;
         else if (argument < 0.0  &&  S_threshold < -argument)
            new_beta = (argument + S_threshold) / update_factor ;
         else
            new_beta = 0.0 ;

         // Apply the update, if changed, and adjust the residual if using naive or weighted updates
         // This is also used to update the fast convergence criterion

         correction = new_beta - beta[ivar] ;  // Will use this to adjust residual if using naive updates
         if (fabs(correction) > max_change)
            max_change = fabs(correction) ;    // Used for fast convergence criterion

         if (correction != 0.0) {   // Did this beta change?
            if (! covar_updates) {  // Must we update the residual vector (needed for naive methods)?
               xptr = x + ivar ;    // Point to column of this variable
               for (icase=0 ; icase<ncases ; icase++)            // Update residual for this new beta
                  resid[icase] -= correction * xptr[icase*nvars] ;
               }
            if ((beta[ivar] == 0.0  &&  new_beta != 0.0)  ||  (beta[ivar] != 0.0  &&  new_beta == 0.0))
               active_set_changed = 1 ;
            beta[ivar] = new_beta ;
            }

         } // For all variables; a complete pass

/*
   A pass (complete or active only) through variables has been done.
   If we are using the fast convergence test, it is simple.  But if using the slow method...
     Compute explained variance and criterion; compare to prior for convergence test
     If the user requested the covariance update method, we must compute residuals for these.
*/

      if (fast_test) {            // Quick and simple test
         if (max_change < eps)
            converged = 1 ;
         else
            converged = 0 ;
         }

      else {   // Slow test (change in explained variance) which requires residual
         if (covar_updates) {  // We have until now avoided computing residuals
            for (icase=0 ; icase<ncases ; icase++) {
               xptr = x + icase * nvars ;
               sum = 0.0 ;
               for (ivar=0 ; ivar<nvars ; ivar++)
                  sum += beta[ivar] * xptr[ivar] ; // Cumulate predicted value
               resid[icase] = y[icase] - sum ;     // Residual = true - predicted
               }
            }

         sum = 0.0 ;       // Will cumulate squared error for convergence test
         if (w != NULL) {  // Are the errors of each case weighted differently?
            for (icase=0 ; icase<ncases ; icase++)
               sum += w[icase] * resid[icase] * resid[icase] ;
            crit = sum ;
            }
         else {
            for (i=0 ; i<ncases ; i++)
               sum += resid[i] * resid[i] ;
            crit = sum / ncases ;               // MSE component of optimization criterion
            }

         explained_variance = (YmeanSquare - crit) / YmeanSquare ; // Fraction of Y explained

         penalty = 0.0 ;
         for (i=0 ; i<nvars ; i++)
            penalty += 0.5 * (1.0 - alpha) * beta[i] * beta[i]  +  alpha * fabs(beta[i]) ;
         penalty *= 2.0 * lambda ;           // Regularization component of optimization criterion

         crit += penalty ;                   // This is what we are minimizing

         if (prior_crit - crit < eps)
            converged = 1 ;
         else
            converged = 0 ;

         prior_crit = crit ;
         }

/*
      After doing a complete (all variables) pass, we iterate on only
      the active set (beta != 0) until convergence.  Then we do a complete pass.
      If the active set does not change, we are done:
      If a beta goes from zero to nonzero, by definition the active set changed.
      If a beta goes from nonzero to another nonzero, then this is a theoretical flaw
      in this process.  However, if we just iterated the active set to convergence,
      it is highly unlikely that we would get anything other than a tiny move.
*/

      if (do_active_only) {      // Are we iterating on the active set only?
         if (converged)          // If we converged
            do_active_only = 0 ; // We now do a complete pass
         }

      else {                     // We just did a complete pass (all variables)
         if (converged  &&  ! active_set_changed)
            break ;
         do_active_only = 1 ;    // We now do an active-only pass
         }

      } // Outer loop iterations

/*
   We are done.  Compute and save the explained variance.
   If we did the fast convergence test and covariance updates,
   we must compute the residual in order to get the explained variance.
   Those two options do not require regular residual computation,
   so we don't currently have the residual.
*/

   if (fast_test  &&  covar_updates) {  // Residuals have not been maintained?
      for (icase=0 ; icase<ncases ; icase++) {
         xptr = x + icase * nvars ;
         sum = 0.0 ;
         for (ivar=0 ; ivar<nvars ; ivar++)
            sum += beta[ivar] * xptr[ivar] ;
         resid[icase] = y[icase] - sum ;
         }
      }

   sum = 0.0 ;
   if (w != NULL) {   // Error term of each case weighted differentially?
      for (i=0 ; i<ncases ; i++)
         sum += w[i] * resid[i] * resid[i] ;
      crit = sum ;
      }
   else {
      for (i=0 ; i<ncases ; i++)
         sum += resid[i] * resid[i] ;
      crit = sum / ncases ;               // MSE component of optimization criterion
      }

   explained = (YmeanSquare - crit) / YmeanSquare ;  // This variable is a member of the class
}


/*
------------------------------------------------------------------------------------------

   Compute minimum lambda such that all zero betas remain at zero

------------------------------------------------------------------------------------------
*/

double CoordinateDescent::get_lambda_thresh ( double alpha )
{
   int ivar, icase ;
   double thresh, sum, *xptr ;

   thresh = 0.0 ;
   for (ivar=0 ; ivar<nvars ; ivar++) {
      xptr = x + ivar ;
      sum = 0.0 ;
      if (w != NULL) {
         for (icase=0 ; icase<ncases ; icase++)
            sum += w[icase] * xptr[icase*nvars] * y[icase] ;
         }
      else {
         for (icase=0 ; icase<ncases ; icase++)
            sum += xptr[icase*nvars] * y[icase] ;
         sum /= ncases ;
         }
      sum = fabs(sum) ;
      if (sum > thresh)
         thresh = sum ;
      }

   return thresh / (alpha + 1.e-60) ;
}


/*
------------------------------------------------------------------------------------------

   Multiple-lambda training routine calls core_train() repeatedly, saving each beta vector

------------------------------------------------------------------------------------------
*/

void CoordinateDescent::lambda_train (
   double alpha ,      // User-specified alpha, (0,1) (0 problematic for descending lambda)
   int maxits ,        // Maximum iterations, for safety only
   double eps ,        // Convergence criterion, typically 1.e-5 or so
   int fast_test ,     // Base convergence on max beta change vs explained variance?
   double max_lambda , // Starting lambda, or negative for automatic computation
   int print_steps     // Print lambda/explained table?
   )
{
   int ivar, ilambda, n_active ;
   double lambda, min_lambda, lambda_factor ;
   FILE *fp_results ;

   if (print_steps) {
      fopen_s ( &fp_results , "CDtest.LOG" , "at" ) ;
      fprintf ( fp_results , "\n\nDescending lambda training..." ) ;
      fclose ( fp_results ) ;
      }

   if (n_lambda <= 1)
      return ;

/*
   Compute the minimum lambda for which all beta weights remain at zero
   This (slightly decreased) will be the lambda from which we start our descent.
*/

   if (max_lambda <= 0.0)
      max_lambda = 0.999 * get_lambda_thresh ( alpha ) ;
   min_lambda = 0.001 * max_lambda ;
   lambda_factor = exp ( log ( min_lambda / max_lambda ) / (n_lambda-1) ) ;

/*
   Repeatedly train with decreasing lambdas
*/

   if (print_steps) {
      fopen_s ( &fp_results , "CDtest.LOG" , "at" ) ;
      fprintf ( fp_results , "\nLambda  n_active  Explained" ) ;
      }

   lambda = max_lambda ;
   for (ilambda=0 ; ilambda<n_lambda ; ilambda++) {
      lambdas[ilambda] = lambda ;   // Save in case we want to use later
      core_train ( alpha , lambda , maxits , eps , fast_test , ilambda ) ;
      for (ivar=0 ; ivar<nvars ; ivar++)
         lambda_beta[ilambda*nvars+ivar] = beta[ivar] ;
      if (print_steps) {
         n_active = 0 ;
         for (ivar=0 ; ivar<nvars ; ivar++) {
            if (fabs(beta[ivar]) > 0.0)
              ++n_active ;
            }
         fprintf ( fp_results , "\n%8.4lf %4d %12.4lf", lambda, n_active, explained ) ;
         }
      lambda *= lambda_factor ;
      }

   if (print_steps)
      fclose ( fp_results ) ;
}


/*
------------------------------------------------------------------------------------------

   Cross-validation training routine calls lambda_train() repeatedly to optimize lambda

------------------------------------------------------------------------------------------
*/

double cv_train (
   int n ,              // Number of cases in full database
   int nvars ,          // Number of variables (columns in database)
   int nfolds ,         // Number of folds
   double *xx ,         // Full database (n rows, nvars columns)
   double *yy ,         // Predicted variable vector, n long
   double *ww ,         // Optional weights, n long, or NULL if no differential weighting
   double *lambdas ,    // Returns lambdas tested by lambda_train()
   double *lambda_OOS , // Returns OOS explained for each of above lambdas
   double *work ,       // Work vector n long
   int covar_updates ,  // Does user want (usually faster) covariance update method?
   int n_lambda ,       // This many lambdas tested by lambda_train() (must be at least 2)
   double alpha ,       // User-specified alpha, (0,1) (0 problematic for descending lambda)
   int maxits ,         // Maximum iterations, for safety only
   double eps ,         // Convergence criterion, typically 1.e-5 or so
   int fast_test        // Base convergence on max beta change vs explained variance?
   )
{
   int i_IS, n_IS, i_OOS, n_OOS, n_done, ifold  ;
   int icase, ivar, ilambda, ibest, k ;
   double pred, sum, diff, *coefs, max_lambda, Ynormalized, YsumSquares, best ;
   CoordinateDescent *cd ;

   if (n_lambda < 2)
      return 0.0 ;

/*
   Use the entire dataset to find the max lambda that will be used for all descents.
   Also, copy the normalized case weights if there are any.
*/

   cd = new CoordinateDescent ( nvars , n , (ww != NULL) , covar_updates , n_lambda ) ;
   cd->get_data ( 0 , n , xx , yy , ww ) ;                     // Fetch the training set for this fold
   max_lambda = cd->get_lambda_thresh ( alpha ) ;
   if (ww != NULL) {
      for (icase=0 ; icase<n ; icase++)
         work[icase] = cd->w[icase] ;
      }
   delete cd ;

#if RESULTS
   FILE *fp_results ;
   fopen_s ( &fp_results , "CDtest.LOG" , "at" ) ;
   fprintf ( fp_results , "\n\n\ncv_train() starting for %d folds with max lambda=%.4lf\n" , nfolds, max_lambda ) ;
   fclose ( fp_results ) ;
#endif

   i_IS = 0 ;        // Training data starts at this index in complete database
   n_done = 0 ;      // Number of cases treated as OOS so far

   for (ilambda=0 ; ilambda<n_lambda ; ilambda++)
      lambda_OOS[ilambda] = 0.0 ;  // Will cumulate across folds here

   YsumSquares = 0.0 ;    // Will cumulate to compute explained fraction

/*
   Process the folds
*/

   for (ifold=0 ; ifold<nfolds ; ifold++) {

      n_OOS = (n - n_done) / (nfolds - ifold) ;  // Number OOS  (test set)
      n_IS = n - n_OOS ;                         // Number IS (training set)
      i_OOS = (i_IS + n_IS) % n ;                // OOS starts at this index

      // Train the model with this IS set

      cd = new CoordinateDescent ( nvars , n_IS , (ww != NULL) , covar_updates , n_lambda ) ;
      cd->get_data ( i_IS , n , xx , yy , ww ) ;                  // Fetch the training set for this fold
      cd->lambda_train ( alpha , maxits , eps , fast_test , max_lambda , 0 ) ; // Compute the complete set of betas (all lambdas)

      // Compute OOS performance for each lambda and sum across folds.
      // Normalization of X and Y is repeated, when it could be done once and saved.
      // But the relative cost is minimal, and it is simpler doing it this way.

      for (ilambda=0 ; ilambda<n_lambda ; ilambda++) {
         lambdas[ilambda] = cd->lambdas[ilambda] ;  // This will be the same for all folds
         coefs = cd->lambda_beta + ilambda * nvars ;
         sum = 0.0 ;
         for (icase=0 ; icase<n_OOS ; icase++) {
            k = (icase + i_OOS) % n ;
            pred = 0.0 ;
            for (ivar=0 ; ivar<nvars ; ivar++)
               pred += coefs[ivar] * (xx[k*nvars+ivar] - cd->Xmeans[ivar]) / cd->Xscales[ivar] ;
            Ynormalized = (yy[k] - cd->Ymean) / cd->Yscale ;
            diff = Ynormalized - pred ;
            if (ww != NULL) {
               if (ilambda == 0)
                  YsumSquares += work[k] * Ynormalized * Ynormalized ;
               sum += work[k] * diff * diff ;
               }
            else {
               if (ilambda == 0)
                  YsumSquares += Ynormalized * Ynormalized ;
               sum += diff * diff ;
               }
            }
         lambda_OOS[ilambda] += sum ;  // Cumulate for this fold
         }  // For ilambda

      delete cd ;

      n_done += n_OOS ;                           // Cumulate OOS cases just processed
      i_IS = (i_IS + n_OOS) % n ;                 // Next IS starts at this index

      }  // For ifold

/*
   Compute OOS explained variance for each lambda, and keep track of the best
*/

   best = -1.e60 ;
   for (ilambda=0 ; ilambda<n_lambda ; ilambda++) {
      lambda_OOS[ilambda] = (YsumSquares - lambda_OOS[ilambda]) / YsumSquares ;
      if (lambda_OOS[ilambda] > best) {
         best = lambda_OOS[ilambda] ;
         ibest = ilambda ;
         }
      }

#if RESULTS
   fopen_s ( &fp_results , "CDtest.LOG" , "at" ) ;
   fprintf ( fp_results , "\ncv_train() ending with best lambda=%.4lf  explained=%.4lf", lambdas[ibest], best ) ;
   fclose ( fp_results ) ;
#endif

   return lambdas[ibest] ;
}
