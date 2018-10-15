/******************************************************************************/
/*                                                                            */
/*  PARAMCOR - Compute and print parameter correlation info.                  */
/*             This is called from DIFF_EV.CPP.                               */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "headers.h"

int paramcor (
   int ncases ,     // Number of cases
   int nparams ,    // Number of parameters
   double *data     // Ncases (rows) by nparams+1 input of trial pts and f vals
   )
{
   int i, j, k, ibest, ncoefs, *iwork, nc_kept ;
   double *aptr, *cptr, *pptr, *coefs, *hessian, *evals, *evect, *work1, *dwork ;
   double *best, sum, diff, d, d2, limit, corr, lscale, rscale, best_val, *bestptr ;
   char msg[1024], msg2[256] ;
   FILE *fp ;
   SingularValueDecomp *sptr ;

   if (nparams < 2)
      return 1 ;

   ncoefs = nparams                       // First-order terms
          + nparams * (nparams + 1) / 2   // Second-order terms
          + 1 ;                           // Constant

   // The multiplier to get nc_kept is a tradeoff.
   // It should be small so that we stay near the best individual and get local behavior.
   // (Distant individuals just confuse our estimates of behavior near the best.)
   // But it should be large enough to capture all possible parameter interactions.
   // My choice of 1.5 is totally heuristic.  I would not go any smaller,
   // but larger values are reasonable.

   nc_kept = (int) (1.5 * ncoefs) ;  // Keep this many individuals

   if (nc_kept > ncases)
      nc_kept = ncases ;

/*
   Do all allocation at once so we can just abort in the unlikely event that there is a problem
*/

   sptr = new SingularValueDecomp ( nc_kept , ncoefs , 0 ) ;
   coefs = (double *) malloc ( ncoefs * sizeof(double) ) ;
   hessian = (double *) malloc ( nparams * nparams * sizeof(double) ) ;
   evals = (double *) malloc ( nparams * sizeof(double) ) ;
   evect = (double *) malloc ( nparams * nparams * sizeof(double) ) ;
   work1 = (double *) malloc ( nparams * sizeof(double) ) ;
   dwork = (double *) malloc ( ncases * sizeof(double) ) ;
   iwork = (int *) malloc ( ncases * sizeof(int) ) ;
   if (fopen_s ( &fp , "PARAMCOR.LOG" , "wt" ))
      fp = NULL ;

   if ((sptr == NULL)  ||  (! sptr->ok)  ||  (coefs == NULL)
    || (hessian == NULL)  || (evals == NULL)  || (evect == NULL)
    || (work1 == NULL)  || (dwork == NULL)  || (iwork == NULL)  || (fp == NULL)) {
      if (sptr != NULL)
         delete sptr ;
      if (coefs != NULL)
         free ( coefs ) ;
      if (hessian != NULL)
         free ( hessian ) ;
      if (evals != NULL)
         free ( evals ) ;
      if (evect != NULL)
         free ( evect ) ;
      if (work1 != NULL)
         free ( work1 ) ;
      if (dwork != NULL)
         free ( dwork ) ;
      if (iwork != NULL)
         free ( iwork ) ;
      if (fp != NULL)
         fclose ( fp ) ;
      return 1 ;
      }

/*
   Find the best individual.
   This is partly for numerical stability, but mainly so we can gather individuals
   whose parameters values are near it so we can model local behavior.
*/

   for (i=0 ; i<ncases ; i++) {
      pptr = data + i * (nparams+1) ;
      if (i==0  ||  pptr[nparams] > best_val) {
         ibest = i ;
         best_val = pptr[nparams] ;
         }
      }

   bestptr = data + ibest * (nparams+1) ;   // This is the best individual

   // Compute the distance from the best for each individual
   // Then sort them to get the indices of the sorted individuals

   for (i=0 ; i<ncases ; i++) {
      pptr = data + i * (nparams+1) ;
      sum = 0.0 ;
      for (j=0 ; j<nparams ; j++) {
         diff = pptr[j] - bestptr[j] ;
         sum += diff * diff ;
         }
      dwork[i] = sum ;
      iwork[i] = i ;
      }

   qsortdsi ( 0 , ncases-1 , dwork , iwork ) ; // Closest to most distant

/*
   Place the closest parameter trials in 'a' and their corresponding function
   values in 'b' and then solve for the coefficients.
   To aid numerical stability, we subtract the best (params and fval) from
   each.  This is not mathematically necessary because it would be absorbed
   in the constant.  But it helps fpt accuracy.
   Also, we flip the sign of the functions, converting this vicinity from
   a maximum to a minimum.  This encourages the Hessian to be positive
   definite rather than negative definite, which makes diagnostic printouts
   easier to read (fewer negative signs).  It has no mathematical effect.
   From this point on, we will refer to this as a minimum for clarity.
*/

   aptr = sptr->a ;                    // Design matrix goes here
   best = data + ibest * (nparams+1) ; // Best individual, parameters and value

   for (i=0 ; i<nc_kept ; i++) {
      pptr = data + iwork[i] * (nparams+1) ;
      for (j=0 ; j<nparams ; j++) {
         d = pptr[j] - best[j] ;
         *aptr++ = d ;              // First-order terms
         for (k=j ; k<nparams ; k++) {
            d2 = pptr[k] - best[k] ;
            *aptr++ = d * d2 ;   // Second-order terms
            }
         }
      *aptr++ = 1.0 ;  // Constant term
      sptr->b[i] = best[nparams] - pptr[nparams] ;  // RHS is function values
      }

   sptr->svdcmp () ;
   sptr->backsub ( 1.e-10 , coefs ) ; // Computes optimal weights

   fprintf ( fp , "Coefficients fitting performance to parameters, linear first, then quadratic, then mixed\n" ) ;
   cptr = coefs ;
   for (j=0 ; j<nparams ; j++) {
      sprintf_s ( msg , "%11.3le :", *cptr ) ;
      ++cptr ;
      for (k=j ; k<nparams ; k++) {
         sprintf_s ( msg2 , " %11.3le", *cptr ) ;
         strcat_s ( msg , msg2 ) ;
         ++cptr ;
         }
      fprintf ( fp , "\n%s", msg ) ;
      }
   sprintf_s ( msg , "Constant: %.3le", *cptr ) ;
   fprintf ( fp , "\n%s", msg ) ;

   delete sptr ;

/*
   Compute the Hessian matrix.
*/

   cptr = coefs ;
   for (j=0 ; j<nparams ; j++) {
      ++cptr ;   // Skip the linear term
      for (k=j ; k<nparams ; k++) {
         hessian[j*nparams+k] = *cptr ;
         if (k == j)                        // If this is a diagonal element
            hessian[j*nparams+k] *= 2.0 ;   // Second partial is twice coef
         else                               // If off-diagonal
            hessian[k*nparams+j] = *cptr ;  // Copy the symmetric element
         ++cptr ;
         }
      }

   fprintf ( fp , "\n\nHessian before adjustment\n" ) ;
   for (j=0 ; j<nparams ; j++) {
      strcpy_s ( msg , "" ) ;
      for (k=0 ; k<nparams ; k++) {
         sprintf_s ( msg2 , " %11.3le", hessian[j*nparams+k] ) ;
         strcat_s ( msg , msg2 ) ;
         }
      fprintf ( fp , "\n%s", msg ) ;
      }

/*
   If we are in the vicinity of a true minimum, all diagonals will be positive.
   But that may not happen in real life.  For any diagonals that are not
   now reasonably positive, zero them and their row and column.
   It's sad doing this, but there is no point in talking about level ellipses
   when we are at a saddle point.
*/

   for (j=0 ; j<nparams ; j++) {
      if (hessian[j*nparams+j] < 1.e-10) {
         for (k=j ; k<nparams ; k++)
            hessian[j*nparams+k] = hessian[k*nparams+j] = 0.0 ;
         }
      }

/*
   Again, if we are in the vicinity of a true minimum, the Hessian matrix
   will be positive semi-definite.  Encourage this if needed.
   Weird correlation patterns may still give us one or more negative eigenvalues.
   But we remain optimistic and impose a necessary but not sufficient condition.
*/

   for (j=0 ; j<nparams-1 ; j++) {
      d = hessian[j*nparams+j] ;       // One diagonal
      for (k=j+1 ; k<nparams ; k++) {
         d2 = hessian[k*nparams+k] ;   // Another diagonal
         limit = 0.99999 * sqrt ( d * d2 ) ;
         if (hessian[j*nparams+k] > limit) {
            hessian[j*nparams+k] = limit ;
            hessian[k*nparams+j] = limit ;
            }
         if (hessian[j*nparams+k] < -limit) {
            hessian[j*nparams+k] = -limit ;
            hessian[k*nparams+j] = -limit ;
            }
         }
      }

   fprintf ( fp , "\n\nHessian after adjustment to encourage nonnegative eigenvalues\n" ) ;
   for (j=0 ; j<nparams ; j++) {
      strcpy_s ( msg , "" ) ;
      for (k=0 ; k<nparams ; k++) {
         sprintf_s ( msg2 , " %11.3le", hessian[j*nparams+k] ) ;
         strcat_s ( msg , msg2 ) ;
         }
      fprintf ( fp , "\n%s", msg ) ;
      }

/*
   Compute eigenstructure of Hessian.
*/

   evec_rs ( hessian , nparams , 1 , evect , evals , work1 ) ;

   fprintf ( fp , "\n\nEigenvalues (top row) with corresponding vectors below each\n" ) ;
   strcpy_s ( msg , "" ) ;
   for (j=0 ; j<nparams ; j++) {
      sprintf_s ( msg2 , " %11.3le", evals[j] ) ;
      strcat_s ( msg , msg2 ) ;
      }
   fprintf ( fp , "\n%s", msg ) ;

   for (j=0 ; j<nparams ; j++) {
      strcpy_s ( msg , "" ) ;
      for (k=0 ; k<nparams ; k++) {
         sprintf_s ( msg2 , " %11.3le", evect[j*nparams+k] ) ;
         strcat_s ( msg , msg2 ) ;
         }
      fprintf ( fp , "\n%s", msg ) ;
      }

/*
   Compute generalized inverse of Hessian.
   Reuse 'hessian' for the inverse.
*/

   for (j=0 ; j<nparams ; j++) {
      for (k=j ; k<nparams ; k++) {
         sum = 0.0 ;
         for (i=0 ; i<nparams ; i++) {
            if (evals[i] > 1.e-8)
               sum += evect[j*nparams+i] * evect[k*nparams+i] / evals[i] ;
            }
         hessian[j*nparams+k] = hessian[k*nparams+j] = sum ;
         }
      }

   fprintf ( fp , "\n\nGeneralized inverse of modified Hessian\n" ) ;
   for (j=0 ; j<nparams ; j++) {
      strcpy_s ( msg , "" ) ;
      for (k=0 ; k<nparams ; k++) {
         sprintf_s ( msg2 , " %11.3le", hessian[j*nparams+k] ) ;
         strcat_s ( msg , msg2 ) ;
         }
      fprintf ( fp , "\n%s", msg ) ;
      }

/*
   Print parameter variation
*/

   fprintf ( fp, "\n\nEstimated parameter variation and correlations\n" ) ;
   fprintf ( fp, "\nVariation very roughly indicates how much the parameter can change" ) ;
   fprintf ( fp, "\nRELATIVE to the others without having a huge impact on performance.\n" ) ;
   fprintf ( fp, "\nA strong positive correlation between A and B means that an increase" ) ;
   fprintf ( fp, "\nin parameter A can be somewhat offset by an increase in parameter B.\n" ) ;
   fprintf ( fp, "\nA strong negative correlation between A and B means that an increase" ) ;
   fprintf ( fp, "\nin parameter A can be somewhat offset by a decrease in parameter B.\n" ) ;

   for (i=0 ; i<nparams ; i++) {          // Scale so largest variation is 1.0
      if (hessian[i*nparams+i] > 0.0)
         d = sqrt ( hessian[i*nparams+i] ) ;
      else
         d = 0.0 ;
      if (i == 0  ||  d > rscale)
         rscale = d ;
      }

   strcpy_s ( msg , "               " ) ;
   for (i=0 ; i<nparams ; i++) {
      sprintf_s ( msg2, "      Param %d", i+1 ) ;
      strcat_s ( msg , msg2 ) ;
      }
   fprintf ( fp , "\n%s", msg ) ;

   strcpy_s ( msg , "  Variation-->" ) ;
   for (i=0 ; i<nparams ; i++) {
      if (hessian[i*nparams+i] > 0.0)
         d = sqrt ( hessian[i*nparams+i] ) / rscale ;
      else
         d = 0.0 ;
      sprintf_s ( msg2 , " %12.3lf", d ) ;
      strcat_s ( msg , msg2 ) ;
      }
   fprintf ( fp , "\n%s", msg ) ;

/*
   Print paramter correlations.
*/

   for (i=0 ; i<nparams ; i++) {
      sprintf_s ( msg, "  %12d", i+1 ) ;
      if (hessian[i*nparams+i] > 0.0)
         d = sqrt ( hessian[i*nparams+i] ) ;
      else
         d = 0.0 ;
      for (k=0 ; k<nparams ; k++) {
         if (hessian[k*nparams+k] > 0.0)
            d2 = sqrt ( hessian[k*nparams+k] ) ;
         else
            d2 = 0.0 ;
         if (d * d2 > 0.0) {
            corr = hessian[i*nparams+k] / (d * d2) ;
            if (corr > 1.0)
               corr = 1.0 ;
            if (corr < -1.0)
               corr = -1.0 ;
            sprintf_s ( msg2 , " %12.3lf", corr ) ;
            }
         else
            strcpy_s ( msg2 , "        -----" ) ;
         strcat_s ( msg , msg2 ) ;
         }
      fprintf ( fp , "\n%s", msg ) ;
      }

/*
   If there are at least two positive eigenvalues,
   print min and max sensitivity vectors.
   Moving in the direction of maximum sensitivity causes the most
   change in the performance.  Moving in the minimum direction
   causes the least change in performance.
*/

   if (nparams < 2)
      goto FINISH ;

   for (k=nparams-1 ; k>0 ; k--) { // Find the smallest positive eigenvalue
      if (evals[k] > 0.0)
         break ;
      }

   if (! k)
      goto FINISH ;

   fprintf ( fp, "\n\nDirections of maximum and minimum sensitivity" ) ;
   fprintf ( fp, "\nMoving in the direction of maximum sensitivity causes the most change in performance." ) ;
   fprintf ( fp, "\nMoving in the direction of minimum sensitivity causes the least change in performance.\n" ) ;
   fprintf ( fp, "\n                     Max        Min\n" ) ;

   lscale = rscale = 0.0 ;  // Scale so largest element is 1.0.  Purely heuristic.

   for (i=0 ; i<nparams ; i++) {
      if (fabs ( evect[i*nparams] ) > lscale)
         lscale = fabs ( evect[i*nparams] ) ;
      if (fabs ( evect[i*nparams+k] ) > rscale)
         rscale = fabs ( evect[i*nparams+k] ) ;
      }

   for (i=0 ; i<nparams ; i++) {
      sprintf_s ( msg, "       Param %d %10.3lf %10.3lf",
         i+1, evect[i*nparams] / lscale, evect[i*nparams+k] / rscale) ;
      fprintf ( fp , "\n%s", msg ) ;
      }

FINISH:
   free ( coefs ) ;
   free ( hessian ) ;
   free ( evals ) ;
   free ( evect ) ;
   free ( work1 ) ;
   free ( dwork ) ;
   free ( iwork ) ;
   fclose ( fp ) ;

   return 0 ;
}