/****************************************************************************/
/*                                                                          */
/*                           EVEC_RS                                        */
/*                                                                          */
/*   Compute eigenvalues and vectors of real symmetric matrix               */
/*                                                                          */
/****************************************************************************/
/*                                                                          */

#include <math.h>
#include "headers.h"

/*
   The input matrix is mat_in.  It is not touched.  The upper minor triangle
   of it is ignored, and hence may be garbage.  Its column dimension is n.
   The vect matrix must be supplied, even if the eigenvectors are not computed.
   The eigenvectors, if computed, are output in vect, which has column dimension n.
   The calling program may use the same matrix for mat_in and vect,
   in which case the input is simply replaced.
   The eigenvalues are output in eval.  Workv is a double work vector n long.
   This returns the number of eigenvalues which could not be computed,
   which is virtually always 0.  I've exhaustively tested this routine and
   never seen it return a nonzero value.
*/

int evec_rs ( double *mat_in , int n , int find_vec , double *vect , double *eval , double *workv )
{
   int i, im1, j, k, irow, irowm1, ival, ivalp1, iercnt, msplit, ibig ;
   double b, f, g, h, hh, p, r, x, scale, shift, sine, cosine, big, *vptr ;

   // Compzero is an accuracy versus speed tradeoff.  The algorithm is most accurate when compzero=0.
   // But by letting 'zero' be a very small positive number, we can take some early loop exits
   // with very little penalty, insignificant most of the time.
   double compzero = 1.e-16 ;
//   double compzero = 0.0 ;

   // Eps is used only for splitting a large matrix into two smaller matrices at a 'zero' diagonal,
   // greatly speeding operation.  But if the diagonal is not quite zero, this does introduce a tiny,
   // usually insignificant, error.
   // The algorithm is most accurate when eps=0, but very small values are fine for most work.
   double eps = 1.e-12 ;
//   double eps = 0.0 ;
   
   /* copy lower triangle of input to output. */
   for (i=0 ; i<n ; i++) {
      for (j=0 ; j<=i ; j++)
         vect[i*n+j] = mat_in[i*n+j] ;
      }
/*
------------------------------------------------------------------------------

   This section converts the matrix (now in vect) to tri-diagonal form
   using Householder's method.  It is done backwards; The last row is done
   first. The subdiagonal is saved in workv as it is found.

------------------------------------------------------------------------------
*/
   for (irow=n-1 ; irow>0 ; irow--) {
      irowm1 = irow - 1 ;
      h = 0.0 ;
      /* We can improve computational accuracy by scaling the row. */
      for (scale=0.0 , i=0 ; i<=irowm1 ; i++) /* do left of diag only */
         scale += fabs ( vect[irow*n+i] ) ;
      /* Avoid a lot of work if this row already tri-diagonal */
      if (scale < compzero  ||  irow == 1)
         workv[irow] = vect[irow*n+irowm1] ;
      else {
         /*  Do actual scaling (left of diag only).  Cumulate sum squares */
         for (i=0 ; i<=irowm1 ; i++) {
            x = vect[irow*n+i] / scale ;
            vect[irow*n+i] = x ;
            h += x * x ;
            }
         /*  The 'U' vector of the literature is the row vector except that
             its first element (f) has the length of the vector (sqrt(h))
             either added or subtracted (g), whichever gives the largest
             absolute value. */
         f = vect[irow*n+irowm1] ;
         g = ( f > 0 )  ?  -sqrt (h)  :  sqrt (h)  ;
         workv[irow] = g * scale ;  /* subdiagonal compensated for scaling */

         h -= f * g ;
         vect[irow*n+irowm1] = f - g ;

         /* Prepare to reduce vect.  Use upper triangle for storage. */
         
         for (f=0.0 , j=0 ; j<=irowm1 ; j++) {
            if (find_vec)
               vect[j*n+irow] = vect[irow*n+j] / h ;
            /* Form element of A * U */
            for (g=0.0 , k=0 ; k<=j ; k++)
               g += vect[j*n+k] * vect[irow*n+k] ;
            if (j < irowm1)
               for (k=j+1 ; k<=irowm1 ; k++)
                  g += vect[k*n+j] * vect[irow*n+k] ;
            /* Compute an element of P.  Use the positions in workv below
               those already determined subdiagonals as work areas. */
            workv[j] = g / h ;
            f += workv[j] * vect[irow*n+j] ;
            }  /* for f=0.0  j=0  */

         /* Reduce A such that all elements of row irow are zero except the
            diagonal and the element to its left (ignoring symmetric
            elements).  Naturally we need not compute those zeroes.  Just
            modify the rows above irow.  */
         hh = f / (h + h) ;
         for (j=0 ; j<=irowm1 ; j++) {
            f = vect[irow*n+j] ;
            g = workv[j] - hh * f ;
            workv[j] = g ;
            for (k=0 ; k<=j ; k++)
               vect[j*n+k] -= f * workv[k] + g * vect[irow*n+k] ;
            }
         }  /*  else scale<compzero  */

      /* We are done with this row!  Save h in eval.  */
      eval[irow] = h ;
      }  /* for irow=n-1 */
/*
------------------------------------------------------------------------------

   We are nearly done with the tri-diagonalization.  The transformation
   itself has been done to the matrix and the subdiagonals are stored in  
   workv.  H for each row is in eval.  Complete the job by recovering
   the transformation matrix and diagonal.

------------------------------------------------------------------------------
*/
   workv[0] = 0.0 ;
   if (find_vec) {
      eval[0] = 0.0 ;
      for (irow=0 ; irow<n ; irow++) {
         irowm1 = irow-1  ;  /* following if insures no negative subscript! */
         if (fabs (eval[irow])  >  compzero) {
            for (j=0 ; j<=irowm1 ; j++) {
               for (g=0.0 , k=0 ; k<=irowm1 ; k++)
                  g += vect[irow*n+k] * vect[k*n+j] ;
               for (k=0 ; k<=irowm1 ; k++)
                  vect[k*n+j] -= g * vect[k*n+irow] ;
               }
            }
         /*  Recover diagonal and zero matrix elements which are truly zero
             but were not computed.  */
         eval[irow] = vect[irow*n+irow] ;
         vect[irow*n+irow] = 1. ;
         for (j=0 ; j<=irowm1 ; j++) {
            vect[irow*n+j] = 0.0 ;
            vect[j*n+irow] = 0.0 ;
            }
         }  /*  for  irow=0  */
      } // If find_vec

   else {
      for (irow=0 ; irow<n ; irow++)
         eval[irow] = vect[irow*n+irow] ;
      }   

/*
------------------------------------------------------------------------------

   The matrix is now completely tridiagonal.  The diagonal is in eval and
   the subdiagonal still in workv.  The transformation matrix is in vect. 
   Now we use the QL method to find the eigenvalues and vectors.

------------------------------------------------------------------------------
*/
   if (n == 1)
      return ( 0 ) ;

   /*  The first element of the subdiagonal does not exist.  Shift workv.  */
   for (i=1 ; i<n ; i++)
      workv[i-1] = workv[i] ;
   workv[n-1] = 0.0 ;

   shift = 0.0 ;
   b = 0.0 ;
   /*
      This is the main loop.  The rotation isolates one eigenvalue at a time.
   */
   for (ival=0 ; ival<n ; ival++) {
      iercnt = 0 ;  /* count tries for this eigenvalue  */
      /*  It is always nice to be able to split a matrix into two parts
          in order to reduce it from one big problem to two smaller ones.
          We use 'b' as a computational zero.  If a subdiagonal element
          is smaller than b we have a split.  */
      h = eps * ( fabs (eval[ival]) + fabs (workv[ival] ) ) ;
      h = (h > compzero) ? h : compzero ;  /* needed in some cases */
      b = (b > h) ? b : h  ;
      /* Recall we set workv[n-1]=0.0  This loop at least finds that.  */
      for (msplit=ival ; msplit<n ; msplit++)
         if (fabs ( workv[msplit] ) <= b)
            break ;

      /*  We might luck out.  If the first subdiagonal is 'zero' then
          the corresponding diagonal is an eigenvalue.  Thus we only need to
          do the computation if that is not the case.  */
      if ( msplit > ival) {
         do {
            if (iercnt++ > 100)  /* avoid useless repetition */
               return (n - ival) ;
            /*  Before transforming we shift all eigenvalues by a constant to
                accelerate convergence.  Now shift by an additional h for
                this one.  */
            ivalp1 = ival + 1 ;
            g = eval[ival] ;
            p = ( eval[ivalp1] - g )  /  (2. * workv[ival]);/* tricky denom */
            r = sqrt ( p * p + 1.0 ) ;
            eval[ival] = workv[ival] / ( p + ( (p>0) ? r : -r ) )  ;

            h = g - eval[ival] ;
            /* We just shifted ival'th.  Do same for others.  */
            for (i=ivalp1 ; i<n ; i++)  /* above 'if' insures ivalp1<n */
               eval[i] -= h ;
            shift += h ;
            /* This is the actual QL transform */
            p = eval[msplit] ;
            cosine = 1.0 ;
            sine = 0.0 ;
            /* Only rotate between last eigenvalue computed and split point */
            for (i=msplit-1 ; i >= ival ; i--) {
               g = cosine * workv[i] ;
               h = cosine * p ;
               if (fabs (p) >= fabs (workv[i])) {
                  cosine = workv[i] / p ;
                  r = sqrt ( cosine * cosine + 1.0 ) ;
                  workv[i+1] = sine * p * r ;
                  sine = cosine / r ;
                  cosine = 1.0 / r ;
                  }
               else {
                  cosine = p / workv[i] ;
                  r = sqrt ( cosine * cosine + 1.0 ) ;
                  workv[i+1] = sine * workv[i] * r ;
                  sine = 1.0 / r ;
                  cosine = cosine * sine ;
                  }
               p = cosine * eval[i] - sine * g ;
               eval[i+1] = h + sine * (cosine * g + sine * eval[i]) ;
               /* now we must transform vect the same way, so that we get
                  the eigenvector of the original matrix.  Note that
                  previous vectors are untouched.  */
               if (find_vec) {
                  for (k=0 ; k<n ; k++) {
                     vptr = vect + k * n + i ;
                     h = vptr[1] ;
                     vptr[1] = sine * *vptr  +  cosine * h ;
                     *vptr = cosine * *vptr  -  sine * h ;
                     }
                  }
               }  /*  for i=msplit-1  */
            /*  A tentative eigenvalue has been found.  Save it.  */
            eval[ival] = cosine * p ;
            workv[ival] = sine * p ;

            /*  Repeat until satisfactory accuracy is achieved.  */
            } while ( fabs (workv[ival])  > b ) ;
         }  /*  if  msplit > ival  */
      /*  We have an eigenvalue.  Compensate for shifting.  */
      eval[ival] += shift ;
      }  /*  for ival=0  */
 /*
------------------------------------------------------------------------------

   This is it.  We are all done.  However, many programs prefer for the  
   eigenvalues (and corresponding vectors!) to be sorted in decreasing    
   order.  Do this now.  Then flip signs in any column which has more
   negatives than positives.  This is appreciated during interpretation.

------------------------------------------------------------------------------
*/

   for (i=1 ; i<n ; i++) {
      im1 = i - 1 ;
      ibig = im1 ;
      big = eval[im1] ;
      /*  Find largest eval beyond im1  */
      for (j=i ; j<n ; j++) {
         x = eval[j] ;
         if (x > big) {
            big = x ;
            ibig = j ;
            }
         }
      if (ibig != im1) {
         /* swap */
         eval[ibig] = eval[im1] ;
         eval[im1] = big ;
         if (find_vec) {
            for (j=0 ; j<n ; j++) {
               x = vect[j*n+im1] ;
               p = vect[j*n+ibig] ;  /* using p due to compiler error */
               vect[j*n+im1] = p ;
               vect[j*n+ibig] = x ;
               }
            }
         }
      }

   if (find_vec) {
      for (i=0 ; i<n ; i++) {
         for (k=0 , j=0 ; j<n ; j++)
            if (vect[j*n+i] < 0.)
               k++ ;
         if (2*k > n)
            for (j=0 ; j<n ; j++)
               vect[j*n+i] *= -1. ;
         }
      }
   return ( 0 ) ;
}