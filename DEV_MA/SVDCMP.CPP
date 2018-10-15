/******************************************************************************/
/*                                                                            */
/*  SVDCMP  -  SingularValueDecomp class for computing the singular value     */
/*             decomposition of a rectangular matrix having at least as many  */
/*             rows as columns.                                               */
/*             This also includes a back-substitution routine for computing   */
/*             solutions to linear systems.                                   */
/*                                                                            */
/*  This is based on the implementation in Press "Numerical Recipes."         */
/*                                                                            */
/******************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <malloc.h>
#include <new.h>
#include <float.h>
#include "headers.h"


/*
--------------------------------------------------------------------------------

   SingularValueDecomp - Singular value decomposition

   The following steps are needed to compute a least-squares solution
   to a (possibly overdetermined) linear system:
     1) Create a SingularValueDecomp object.  The constructor will allocate
        memory for the design matrix 'a', the right-hand-side 'b', and all
        scratch memory that it needs.  Optionally, the user can flag the
        constructor to preserve 'a' and return the decomposition in 'u'.
        Normally, 'a' is overwritten.
     2) The design matrix must be placed in 'a' and svdcmp called.
     3) Place the right-hand-side in 'b'
     4) Allocate a vector where the solution is to be placed.
        Call backsub with a pointer to this vector.

--------------------------------------------------------------------------------
*/

inline double root_ss ( double x , double y )
{
   double ratio ;
   if (x < 0.0)
      x = -x ;
   if (y < 0.0)
      y = -y ;
   if (x > y) {
      ratio = y / x ;
      return x * sqrt ( ratio * ratio + 1.0 ) ;
      }
   else if (y == 0.0)
      return 0.0 ;
   else {
      ratio = x / y ;
      return y * sqrt ( ratio * ratio + 1.0 ) ;
      }
}

/*
--------------------------------------------------------------------------------

   Constructor - Allocate input/output and scratch memory.
                 Normally, this returns ok=1.  If not, the user called it with
                 more columns than rows, or there was insufficient memory.

--------------------------------------------------------------------------------
*/

SingularValueDecomp::SingularValueDecomp ( int nr , int nc , int save_a )
{
   if (nc > nr) {             // Illegal
      rows = cols = ok = 0 ;
      return ;
      }

   a = (double *) malloc ( nr * nc * sizeof(double) ) ;
   w = (double *) malloc ( nc * sizeof(double) ) ;
   v = (double *) malloc ( nc * nc * sizeof(double) ) ;
   b = (double *) malloc ( nr * sizeof(double) ) ;
   work = (double *) malloc ( nc * sizeof(double) ) ;
   if (save_a)
      u = (double *) malloc ( nr * nc * sizeof(double) ) ;
   else
      u = NULL ;

   if ((a == NULL)  ||  (w == NULL)  ||  (v == NULL)  ||  (b == NULL)  ||
       (work == NULL)  ||  (save_a && (u == NULL))) {
      if (a != NULL)
         free ( a ) ;
      if (w != NULL)
         free ( w ) ;
      if (v != NULL)
         free ( v ) ;
      if (b != NULL)
         free ( b ) ;
      if (work != NULL)
         free ( work ) ;
      if (u != NULL)
         free ( u ) ;
      rows = cols = ok = 0 ;
      return ;
      }

   ok = 1 ;       // Flag to user that all went well
   rows = nr ;
   cols = nc ;
}

/*
--------------------------------------------------------------------------------

   Destructor - Free memory

--------------------------------------------------------------------------------
*/

SingularValueDecomp::~SingularValueDecomp ()
{
   if (! ok)    // If constructor's mallocs failed
      return ;  // there is nothing to free

   free ( a ) ;
   free ( w ) ;
   free ( v ) ;
   free ( b ) ;
   free ( work ) ;
   if (u != NULL)
      free ( u ) ;
}


/*
--------------------------------------------------------------------------------

   svdcmp - Singular value decomposition of 'a'

--------------------------------------------------------------------------------
*/

void SingularValueDecomp::svdcmp ()
{
   int i, sval, split, iter_limit ;
   double *matrix ;

   if (u != NULL) {   // Must we keep 'a' intact?
      memcpy ( u , a , rows * cols * sizeof(double) ) ;  // If so, copy it
      matrix = u ;                                       // And work on copy
      }
   else              // If not, operate directly on 'a'
      matrix = a ;

   bidiag ( matrix ) ;       // Reduce to bidiagonal
   right ( matrix ) ;        // Accumulate right transforms
   left ( matrix ) ;         // And left

   sval = cols ;
   while (sval--) {    // Loop over the singular values in reverse order
      iter_limit = 50 ;
      while (iter_limit--) {  // Avoid nearly endless loop (very rare!)
         split = sval + 1 ;
         while (--split) {    // Keep splitting as long as possible
            if (norm + fabs (work[split]) == norm) {
               break ;
               }
            if (norm + fabs (w[split-1]) == norm) {
               cancel ( split , sval , matrix ) ;
               break ;
               }
            }
         if (split == sval) {     // Converged?
            if (w[sval] < 0.0) {  // Keep them nonnegative
               w[sval] = -w[sval] ;
               for (i=0 ; i<cols ; i++)
                  v[i*cols+sval] = -v[i*cols+sval] ;
               }
            break ;
            }
         qr ( split , sval , matrix ) ;
         }
      }
}

/*
--------------------------------------------------------------------------------

   bidiag - Householder reduction to bidiagonal

--------------------------------------------------------------------------------
*/

void SingularValueDecomp::bidiag ( double *matrix )
{
   int col, k ;
   double temp, testnorm, scale ;

   norm = temp = scale = 0.0 ;

   for (col=0 ; col<cols ; col++) {

      work[col] = scale * temp ;

      scale = 0.0 ;
      for (k=col ; k<rows ; k++)
         scale += fabs ( matrix[k*cols+col] ) ;

      if (scale > 0.0)
         w[col] = scale * bid1 ( col , matrix , scale ) ;
      else 
         w[col] = 0.0 ;

      scale = 0.0 ;
      for (k=col+1 ; k<cols ; k++)
         scale += fabs ( matrix[col*cols+k] ) ;

      if (scale > 0.0)
         temp = bid2 ( col , matrix , scale ) ;
      else
         temp = 0.0 ;

      testnorm = fabs (w[col]) + fabs (work[col]) ;
      if (testnorm > norm)
         norm = testnorm ;
      }
}

double SingularValueDecomp::bid1 ( int col , double *matrix , double scale )
{
   int i, j ;
   double diag, rv, fac, sum ;

   sum = 0.0 ;
   for (i=col ; i<rows ; i++) {
      fac = (matrix[i*cols+col] /= scale) ;
      sum += fac * fac ;
      }
   rv = sqrt ( sum ) ;
   diag = matrix[col*cols+col] ;
   if (diag > 0.0)
      rv = -rv ;
   fac = 1.0 / (diag * rv - sum) ;
   matrix[col*cols+col] = diag - rv ;

   for (j=col+1 ; j<cols ; j++) {
      sum = 0.0 ;
      for (i=col ; i<rows ; i++)
         sum += matrix[i*cols+col] * matrix[i*cols+j] ;
      sum *= fac ;
      for (i=col ; i<rows ; i++)
         matrix[i*cols+j] += sum * matrix[i*cols+col] ;
      }

   for (i=col ; i<rows ; i++)
      matrix[i*cols+col] *= scale ;

   return rv ;
}

double SingularValueDecomp::bid2 ( int col , double *matrix , double scale )
{
   int i, j ;
   double fac, diag, rv, sum ;

   sum = 0.0 ;
   for (i=col+1 ; i<cols ; i++) {
      fac = (matrix[col*cols+i] /= scale) ;
      sum += fac * fac ;
      }

   rv = sqrt ( sum ) ;
   diag = matrix[col*cols+col+1] ;
   if (diag > 0.0)
      rv = -rv ;

   matrix[col*cols+col+1] = diag - rv ;
   fac = 1.0 / (diag * rv - sum) ;
   for (i=col+1 ; i<cols ; i++)
      work[i] = fac * matrix[col*cols+i] ;

   for (j=col+1 ; j<rows ; j++) {
      sum = 0.0 ;
      for (i=col+1 ; i<cols ; i++)
         sum += matrix[j*cols+i] * matrix[col*cols+i] ;
      for (i=col+1 ; i<cols ; i++)
         matrix[j*cols+i] += sum * work[i] ;
      }
   for (i=col+1 ; i<cols ; i++)
      matrix[col*cols+i] *= scale ;
   return rv ;
}


/*
--------------------------------------------------------------------------------

   Cumulate right and left transforms

--------------------------------------------------------------------------------
*/

void SingularValueDecomp::right ( double *matrix )
{
   int col, i, j ;
   double temp, denom, sum ;

   denom = 0.0 ;
   col = cols ;
   while (col--) {
      if (denom != 0.0) {
         temp = 1.0 / matrix[col*cols+col+1] ;
         for (i=col+1 ; i<cols ; i++)  // Double division avoids underflow
            v[i*cols+col] = temp * matrix[col*cols+i] / denom ;
         for (i=col+1 ; i<cols ; i++) {
            sum = 0.0 ;
            for (j=col+1 ; j<cols ; j++)
               sum += v[j*cols+i] * matrix[col*cols+j] ;
            for (j=col+1 ; j<cols ; j++)
               v[j*cols+i] += sum * v[j*cols+col] ;
            }
         }

      denom = work[col] ;

      for (i=col+1 ; i<cols ; i++)
         v[col*cols+i] = v[i*cols+col] = 0.0 ;
      v[col*cols+col] = 1.0 ;
      }
}

void SingularValueDecomp::left ( double *matrix )
{
   int col, i, j ;
   double temp, fac, sum ;

   col = cols ;
   while (col--) {

      for (i=col+1 ; i<cols ; i++)
         matrix[col*cols+i] = 0.0 ;

      if (w[col] == 0.0) {
         for (i=col ; i<rows ; i++)
            matrix[i*cols+col] = 0.0 ;
         }

      else {
         fac = 1.0 / w[col] ;
         temp = fac / matrix[col*cols+col]  ;

         for (i=col+1 ; i<cols ; i++) {
            sum = 0.0 ;
            for (j=col+1 ; j<rows ; j++)
               sum += matrix[j*cols+col] * matrix[j*cols+i] ;
            sum *= temp ;
            for (j=col ; j<rows ; j++)
               matrix[j*cols+i] += sum * matrix[j*cols+col] ;
            }
         for (i=col ; i<rows ; i++)
            matrix[i*cols+col] *= fac ;
         }

      matrix[col*cols+col] += 1.0 ;
      }
}


/*
--------------------------------------------------------------------------------

   cancel

--------------------------------------------------------------------------------
*/

void SingularValueDecomp::cancel (
   int low ,
   int high ,
   double *matrix
   )
{
   int col, row, lm1 ;
   double sine, cosine, leg1, leg2, svhypot, y, x, *mpt1, *mpt2 ;

   lm1 = low - 1 ;
   sine = 1.0 ;
   for (col=low ; col<=high ; col++) {
      leg1 = sine * work[col] ;
      if (fabs (leg1) + norm != norm) {
         leg2 = w[col] ;
         w[col] = svhypot = root_ss ( leg1 , leg2 ) ;
         sine = -leg1 / svhypot ;
         cosine =  leg2 / svhypot ;
         for (row=0 ; row<rows ; row++) {
            mpt1 = matrix + row * cols + col ;
            mpt2 = matrix + row * cols + lm1 ;
            x = *mpt1 ;
            y = *mpt2 ;
            *mpt1 = x * cosine  -  y * sine ;
            *mpt2 = x * sine  +  y * cosine ;
            }
         }
      }
}

/*
--------------------------------------------------------------------------------

   qr

--------------------------------------------------------------------------------
*/

void SingularValueDecomp::qr (
   int low ,
   int high ,
   double *matrix )
{
   int col ;
   double sine, cosine, wk, tx, ty, x, y, svhypot, temp, ww, wh, wkh, whm1, wkhm1;

   wh = w[high] ;
   whm1 = w[high-1] ;
   wkh = work[high] ;
   wkhm1 = work[high-1] ;
   temp = 2.0 * wkh * whm1 ;
   if (temp != 0.0)
      temp = ((whm1+wh) * (whm1-wh) + (wkhm1+wkh) * (wkhm1-wkh)) / temp ;
   else
      temp = 0.0 ;

   svhypot = root_ss ( temp , 1.0 ) ;
   if (temp < 0.0)
      svhypot = -svhypot ;

   ww = w[low] ;
   wk = wkh * (whm1 / (temp + svhypot) - wkh)  +  (ww+wh) * (ww-wh) ;
   if (ww != 0.0)
      wk /= ww ;
   else
      wk = 0.0 ;

   sine = cosine = 1.0 ;

   for (col=low ; col<high ; col++) {
      x = work[col+1] ;
      ty = sine * x ;
      x *= cosine ;
      svhypot = root_ss ( wk , ty ) ;
      work[col] = svhypot ;
      cosine = wk / svhypot ;
      sine = ty / svhypot ;
      tx = ww * cosine  +  x * sine ;
      x = x * cosine  -  ww * sine ;
      y = w[col+1] ;
      ty = y * sine ;
      y *= cosine ;
      qr_vrot ( col , sine , cosine ) ;
      w[col] = svhypot = root_ss ( tx , ty ) ;
      if (svhypot != 0.0) {
         cosine = tx / svhypot ;
         sine = ty / svhypot ;
         }
      qr_mrot ( col , sine , cosine , matrix ) ;
      wk = cosine * x  +  sine * y ;
      ww = cosine * y  -  sine * x ;
      }
   work[low] = 0.0 ;
   work[high] = wk ;
   w[high] = ww ;
}

void SingularValueDecomp::qr_vrot ( int col , double sine , double cosine )
{
   int row ;
   double x, y, *vptr ;

   for (row=0 ; row<cols ; row++) {
      vptr = v + row * cols + col ;
      x = *vptr ;
      y = *(vptr+1) ;
      *vptr = x * cosine  +  y * sine ;
      *(vptr+1) = y * cosine  -  x * sine ;
      }
}

void SingularValueDecomp::qr_mrot ( int col , double sine , double cosine ,
                                    double *matrix )
{
   int row ;
   double x, y, *mptr ;

   for (row=0 ; row<rows ; row++) {
      mptr = matrix + row * cols + col ;
      x = *mptr ;
      y = *(mptr+1) ;
      *mptr = x * cosine  +  y * sine ;
      *(mptr+1) = y * cosine  -  x * sine ;
      }
}

/*
--------------------------------------------------------------------------------

   Backsubstitution algorithm for solving Ax=b where A generated u, w, v
   Inputs are not destroyed, so it may be called with several b's.
   The user must have filled in the public RHS 'b' before calling this.

--------------------------------------------------------------------------------
*/

void SingularValueDecomp::backsub (
   double limit ,  // SV limit (about sqrt machine precision is good)
   double *soln    // Output: solution
   )
{
   int i, j ;
   double sum, wmax, *matrix ;

   if (u != NULL)    // If we preserved 'a', use 'u'
      matrix = u ;
   else              // Else 'u' is in 'a'
      matrix = a ;

/*
   Scale the threshold to make it relative to the norm
*/

   for (i=0 ; i<cols ; i++) {
      if ((i == 0)  ||  (w[i] > wmax))
         wmax = w[i] ;
      }

   limit = limit * wmax  +  1.e-60 ;

/*
   Find U'b
*/

   for (i=0 ; i<cols ; i++) {
      sum = 0.0 ;
      if (w[i] > limit) {
         for (j=0 ; j<rows ; j++)
            sum += matrix[j*cols+i] * b[j] ;
         sum /= w[i] ;
         }
      work[i] = sum ;
      }

/*
   Multiply by V to complete the solution
*/

   for (i=0 ; i<cols ; i++) {
      sum = 0.0 ;
      for (j=0 ; j<cols ; j++)
         sum += v[i*cols+j] * work[j] ;
      soln[i] = sum ;
      }
}


#if 0
/*
--------------------------------------------------------------------------------

   Optional main to test it

--------------------------------------------------------------------------------
*/

#define RANDMAX 32767

void main ( int argc , char *argv[] )
{
   int rep, m, n, i, j, k, reps ;
   double *x, *sa, sum, err, wmin, wmax ;
   SingularValueDecomp *s ;

   if (argc != 4) {
      printf ( "\nUSAGE: test rows cols reps" ) ;
      exit ( 0 ) ;
      }

   m = atoi ( argv[1] ) ;
   n = atoi ( argv[2] ) ;
   reps = atoi ( argv[3] ) ;

   if (m <= 0  ||  n <= 0  ||  reps <= 0)
      exit ( 0 ) ;

   sa = (double *) malloc ( m * n * sizeof(double) ) ;
   x = (double *) malloc ( n * sizeof(double) ) ;
   s = new SingularValueDecomp ( m , n , 1 ) ;

   if (! s->ok) {
      printf ( "\nError" ) ;
      exit ( 1 ) ;
      }

   for (rep=0 ; rep < reps ; rep++) {

      if (_kbhit()) {
         if (_getch() == 27)
            exit ( 0 ) ;
         }

      if ((m == n)  &&  ! rep) {  // Ill cond
         for (i=0 ; i<m ; i++) {
            for (j=0 ; j<n ; j++)
               sa[i*n+j] = s->a[i*n+j] = 1.0 / (i + j + 1.0) ;
            s->b[i] = (double) (rand() - RANDMAX/2) / (double) RANDMAX ;
            }
         }
      else {
         for (i=0 ; i<m ; i++) {
            for (j=0 ; j<n ; j++) {
               if (j > 100  &&  j % 10 == 0)
                  s->a[i*n+j] = 0.0 ;
               else if (j > 100  &&  j % 10 == 5)
                  s->a[i*n+j] = s->a[i*n+j-1] + s->a[i*n+j-2] ;
               else
                  s->a[i*n+j] = (double) (rand() - RANDMAX/2) / (double) RANDMAX ;
               sa[i*n+j] = s->a[i*n+j] ;
               }
            s->b[i] = (double) (rand() - RANDMAX/2) / (double) RANDMAX ;
            }
         }

      s->svdcmp () ;

      wmin = 1.e30 ;
      wmax = -1.e30 ;
      for (i=0 ; i<n ; i++) {
         if (s->w[i] < wmin)
            wmin = s->w[i] ;
         if (s->w[i] > wmax)
            wmax = s->w[i] ;
         }

      printf ( "\n%d %d (%.2le %.2le)", m, n, wmin, wmax ) ;

      err = 0.0 ;
      for (i=0 ; i<m ; i++) {
         for (j=0 ; j<n ; j++) {
            sum = 0.0 ;
            for (k=0 ; k<n ; k++)
               sum += s->u[i*n+k] * s->w[k] * s->v[j*n+k] ;
            err += fabs ( sum - sa[i*n+j] ) ;
            }
         }

      printf ( " Rep=%.8lf", err ) ;
//      if (fabs(err) > 1.e-10) {
//         printf ( "\a" ) ;
//         _getch() ;
//         }

      err = 0.0 ;
      for (i=0 ; i<n ; i++) {
         for (j=0 ; j<n ; j++) {
            sum = 0.0 ;
            for (k=0 ; k<m ; k++)
               sum += s->u[k*n+i] * s->u[k*n+j] ;
            if (i == j)
               err += fabs ( sum - 1.0 ) ;
            else 
               err += fabs ( sum ) ;
            }
         for (j=0 ; j<n ; j++) {
            sum = 0.0 ;
            for (k=0 ; k<n ; k++)
               sum += s->v[k*n+i] * s->v[k*n+j] ;
            if (i == j)
               err += fabs ( sum - 1.0 ) ;
            else 
               err += fabs ( sum ) ;
            }
         }
      printf ( " Orthog=%.8lf", err ) ;
//      if (fabs(err) > 1.e-10) {
//         printf ( "\a" ) ;
//         _getch() ;
//         }

      if (m == n) {
         s->backsub ( 1.e-8 , x ) ;
         err = 0.0 ;

         for (i=0 ; i<m ; i++) {
            sum = 0.0 ;
            for (j=0 ; j<n ; j++)
               sum += x[j] * sa[i*n+j] ;
            err += fabs ( sum - s->b[i] ) ;
            }

         printf ( " Back=%.8lf", err ) ;
//         if (fabs(err) > 1.e-10) {
//            printf ( "\a" ) ;
//            _getch() ;
//            }
         }

      err = 0.0 ;
      for (i=0 ; i<m ; i++) {
         for (j=0 ; j<n ; j++)
            err += fabs ( sa[i*n+j] - s->a[i*n+j] ) ;
         }
      printf ( " Save=%.8lf", err ) ;
//      if (fabs(err) > 1.e-10) {
//         printf ( "\a" ) ;
//         _getch() ;
//         }
      }

   free ( sa ) ;
   free ( x ) ;
   delete s ;
}
#endif
