/*****************************************************************************/
/*                                                                           */
/*  CD_MA - Send a large number of MA XOVER indicators to a regularized      */
/*          linear model and let it select the best.                         */
/*                                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <conio.h>
#include <assert.h>

#define MKTBUF 2048   /* Alloc for market info in chunks of this many records */
                      /* This is not critical and can be any reasonable vlaue */


/*
---------------------------------------------------------------------------------

   Class declaration

---------------------------------------------------------------------------------
*/

double cv_train ( int n , int nvars , int nfolds , double *xx , double *yy , double *ww ,
                  double *lambdas , double *lambda_OOS , double *work , int covar_updates ,
                  int n_lambda , double alpha , int maxits , double eps , int fast_test ) ;

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
---------------------------------------------------------------------------------

   Local routine computes indicator array for a specified crossover system

---------------------------------------------------------------------------------
*/

void indicators (
   int nind ,        // Number of indicators to compute
   double *x ,       // First case that will be most recent (current)
   int short_term ,  // Short-term lookback
   int long_term ,   // Long-term lookback
   double *inds      // Output of short MA - long MA
   )
{
   int i, j, k ;
   double *xptr, short_mean, long_mean ;

   xptr = x - long_term + 1 ;          // This is the first case examined
   for (i=0 ; i<nind ; i++) {          // For all indicators computed
      k = i + long_term - 1 ;          // Index of current case
      short_mean = 0.0 ;               // Cumulates short-term lookback sum
      for (j=k ; j>k-short_term ; j--)
         short_mean += xptr[j] ;
      long_mean = short_mean ;
      while (j>k-long_term)
         long_mean += xptr[j--] ;
      short_mean /= short_term ;
      long_mean /= long_term ;
      inds[i] = short_mean - long_mean ;
      } // For i, computing indicator array
}


/*
--------------------------------------------------------------------------------

   Main routine

--------------------------------------------------------------------------------
*/

int main (
   int argc ,    // Number of command line arguments (includes prog name)
   char *argv[]  // Arguments (prog name is argv[0])
   )
{
   int i, k, nprices, bufcnt, lookback_inc, n_long, n_short, ivar, nvars, long_lookback, short_lookback ;
   int ilong, ishort, n_train, n_test, n_lambdas, max_lookback ;
   double alpha, pred, sum, *xptr, lambda, *lambdas, *lambda_OOS, *work, *prices, *inds, *targets, *data, *pptr ;
   char line[256], filename[4096], *cptr ;
   FILE *fp, *fp_results ;
   CoordinateDescent *cd ;

/*
   Process command line parameters
*/

#if 1
   if (argc != 6) {
      printf ( "\nUsage: CD_MA  lookback_inc  n_long  n_short  alpha  filename" ) ;
      printf ( "\n  lookback_inc - increment to long-term lookback" ) ;
      printf ( "\n  n_long - Number of long-term lookbacks" ) ;
      printf ( "\n  n_short - Number of short-term lookbacks" ) ;
      printf ( "\n  alpha - Alpha, (0-1]" ) ;
      printf ( "\n  filename - name of market file (YYYYMMDD Price)" ) ;
      exit ( 1 ) ;
      }

   lookback_inc = atoi ( argv[1] ) ;
   n_long = atoi ( argv[2] ) ;
   n_short = atoi ( argv[3] ) ;
   alpha = atof ( argv[4] ) ;
   strcpy_s ( filename , argv[5] ) ;
#else
   lookback_inc = 2 ;
   n_long = 30 ;
   n_short = 10 ;
   alpha = 0.5 ;
   strcpy_s ( filename , "E:\\MarketDataAssorted\\INDEXES\\$OEX.TXT" ) ;
#endif

   if (alpha >= 1.0) {
      printf ( "Alpha must be less than 1.  Press any key..." ) ;
      _getch() ;
      exit ( 1 ) ;
      }

   fopen_s ( &fp_results , "CD_MA.LOG" , "wt" ) ;
   fprintf ( fp_results , "Starting CD_MA with alpha = %.4lf\n", alpha ) ;

/*
   Read market prices
*/

   if (fopen_s ( &fp, filename , "rt" )) {
      printf ( "\n\nCannot open market history file %s", filename ) ;
      exit ( 1 ) ;
      }

   prices = (double *) malloc ( MKTBUF * sizeof(double) ) ;
   if (prices == NULL) {
      printf ( "\n\nInsufficient memory reading market history file %s  Press any key...", filename ) ;
      _getch () ;  // Wait for user to press a key
      fclose ( fp ) ;
      exit ( 1 ) ;
      }

   bufcnt = MKTBUF ;  // There are this many record slots available now

   printf ( "\nReading market file..." ) ;

   nprices = 0 ;    // Counts lines (prices) read

   for (;;) {

      if (feof ( fp )                          // If end of file
       || (fgets ( line , 256 , fp ) == NULL)  // Or unable to read line
       || (strlen ( line ) < 2))               // Or empty line
         break ;                               // We are done reading price history

      if (ferror ( fp )) {                     // If an error reading file
         fclose ( fp ) ;                       // Quit immediately
         free ( prices ) ;
         printf ( "\nError reading line %d of file %s", nprices+1, filename ) ;
         exit ( 1 ) ;
         }

      if (! bufcnt) {  // Allocate a new memory block if needed
         prices = (double *) realloc ( prices , (nprices+MKTBUF) * sizeof(double) ) ;
         if (prices == NULL) {
            fclose ( fp ) ;
            printf ( "\n\nInsufficient memory reading market history file %s  Press any key...", filename ) ;
            _getch () ;  // Wait for user to press a key
            exit ( 1 ) ;
            } // If insufficient memory
         bufcnt = MKTBUF ;  // There are this many new record slots available now
         } // If allocating new block

      // Parse the date and do a crude sanity check

      for (i=0 ; i<8 ; i++) {
         if ((line[i] < '0')  ||  (line[i] > '9')) {
            fclose ( fp ) ;
            free ( prices ) ;
            printf ( "\nInvalid date reading line %d of file %s", nprices+1, filename ) ;
            exit ( 1 ) ;
            }
         }

      // Parse the price

      cptr = line + 9 ;  // Price is in this column or beyond
                         // (Next loop allows price to start past this)

      while (*cptr == ' '  ||  *cptr == '\t'  ||  *cptr == ',')  // Delimiters
         ++cptr ;  // Move up to the price

      prices[nprices] = atof ( cptr ) ;
      if (prices[nprices] > 0.0)                     // Always true, but avoid disaster
         prices[nprices] = log ( prices[nprices] ) ;
      ++nprices  ;
      --bufcnt ;           // One less slot remains

      } // For all lines

   fclose ( fp ) ;

   printf ( "\nMarket price history read" ) ;


/*
   The market data is read.  Initialize for computation.
*/

   n_lambdas = 50 ;
   nvars = n_long * n_short ;
   n_test = 252 ;    // Arbitrary; one year

   max_lookback = n_long * lookback_inc ;
   n_train = nprices - n_test - max_lookback ;  // The last possible current unavailable due to target
   if (n_train < n_long * n_short + 10) {
      free ( prices ) ;
      printf ( "\nERROR... Too little training data for parameters.  Press any key..." ) ;
      _getch () ;  // Wait for user to press a key
      exit ( 1 ) ;
      }

   k = (n_train > n_test)  ?  n_train : n_test ;

   inds = (double *) malloc ( k * sizeof(double) ) ;
   targets = (double *) malloc ( k * sizeof(double) ) ;
   data = (double *) malloc ( k * nvars * sizeof(double) ) ;
   lambdas = (double *) malloc ( n_lambdas * sizeof(double) ) ;
   lambda_OOS = (double *) malloc ( n_lambdas * sizeof(double) ) ;
   work = (double *) malloc ( n_train * sizeof(double) ) ;
   if ((inds == NULL)  ||  (targets == NULL)  ||  (data == NULL)  ||  (lambdas == NULL)  ||  (lambda_OOS == NULL)  ||  (work == NULL)) {
      free ( prices ) ;
      if (inds != NULL)
         free ( inds ) ;
      if (targets != NULL)
         free ( targets ) ;
      if (data != NULL)
         free ( data ) ;
      if (lambdas != NULL)
         free ( lambdas ) ;
      if (lambda_OOS != NULL)
         free ( lambda_OOS ) ;
      if (work != NULL)
         free ( work ) ;
      printf ( "\n\nInsufficient memory.  Press any key..." ) ;
      _getch () ;  // Wait for user to press a key
      exit ( 1 ) ;
      }


/*
   Compute and save indicators for training set
*/

   k = 0 ;
   for (ilong=0 ; ilong<n_long ; ilong++) {
      long_lookback = (ilong+1) * lookback_inc ;
      for (ishort=0 ; ishort<n_short ; ishort++) {
         short_lookback = long_lookback * (ishort+1) / (n_short+1) ;
         if (short_lookback < 1)
            short_lookback = 1 ;
         indicators ( n_train , prices+max_lookback-1 , short_lookback , long_lookback , inds ) ;
         for (i=0 ; i<n_train ; i++)
            data[i*nvars+k] = inds[i] ;
         ++k ;
         }
      }


/*
   Compute and save targets for training set
*/

   pptr = prices + max_lookback - 1 ;  // This is the first current case (indicators use prices through this)
   for (i=0 ; i<n_train ; i++)
      targets[i] = pptr[i+1] - pptr[i] ;


/*
   Compute and print optimal lambda
*/


   if (alpha <= 0.0) {
      alpha = 0.5 ;    // Irrelevant
      lambda = 0.0 ;   // Ordinary linear regression
      fprintf ( fp_results , "\n\nUser specified negative alpha, so lambda = 0" ) ;
      }

   else {
      lambda = cv_train ( n_train , nvars , 10 , data , targets , NULL ,
                  lambdas , lambda_OOS , work , 1 , n_lambdas , alpha , 1000 , 1.e-9 , 1 ) ;
      fprintf ( fp_results , "\n\nCross validation gave optimal lambda = %.4lf  XVAL computation below...", lambda ) ;
      fprintf ( fp_results , "\n  Lambda   OOS explained" ) ;
      for (i=0 ; i<n_lambdas ; i++)
         fprintf ( fp_results , "\n%8.4lf %12.4lf", lambdas[i], lambda_OOS[i] ) ;
      }


/*
   Train the model and print beta coefficients
*/

   cd = new CoordinateDescent ( nvars , n_train , 0 , 1 , 0 ) ;
   cd->get_data ( 0 , n_train , data , targets , NULL ) ;
   cd->core_train ( alpha , lambda , 1000 , 1.e-7 , 1 , 0 ) ;

   fprintf ( fp_results , "\n\nBetas, with in-sample explained variance = %.5lf percent", 100.0 * cd->explained ) ;
   fprintf ( fp_results , "\nRow label is long-term lookback; Columns run from smallest to largest short-term lookback" ) ;

   k = 0 ;
   for (ilong=0 ; ilong<n_long ; ilong++) {
      long_lookback = (ilong+1) * lookback_inc ;
      fprintf ( fp_results , "\n%5d ", long_lookback ) ;
      for (ishort=0 ; ishort<n_short ; ishort++) {
         if (cd->beta[k] != 0.0)
            fprintf ( fp_results , "%9.4lf", cd->beta[k] ) ;
         else
            fprintf ( fp_results , "    ---- " ) ;
         ++k ;
         }
      }

/*
   Compute and save indicators for test set
*/

   k = 0 ;
   for (ilong=0 ; ilong<n_long ; ilong++) {
      long_lookback = (ilong+1) * lookback_inc ;
      for (ishort=0 ; ishort<n_short ; ishort++) {
         short_lookback = long_lookback * (ishort+1) / (n_short+1) ;
         if (short_lookback < 1)
            short_lookback = 1 ;
         indicators ( n_test , prices+n_train+max_lookback-1 , short_lookback , long_lookback , inds ) ;
         for (i=0 ; i<n_test ; i++)
            data[i*nvars+k] = inds[i] ;
         ++k ;
         }
      }


/*
   Compute and save targets for test set
*/

   pptr = prices + n_train + max_lookback - 1 ;  // This is the first current case (indicators use prices through this)
   for (i=0 ; i<n_test ; i++)
      targets[i] = pptr[i+1] - pptr[i] ;


/*
   Do the test
*/

   sum = 0.0 ;
   for (i=0 ; i<n_test ; i++) {
      xptr = data+i*nvars ;
      pred = 0.0 ;
      for (ivar=0 ; ivar<nvars ; ivar++)
         pred += cd->beta[ivar] * (xptr[ivar] - cd->Xmeans[ivar]) / cd->Xscales[ivar] ;
      pred = pred * cd->Yscale + cd->Ymean ; // Unscale prediction to get it back in original Y domain
      if (pred > 0.0)
         sum += targets[i] ;
      else if (pred < 0.0)
         sum -= targets[i] ;
      }

   fprintf ( fp_results , "\n\nOOS total return = %.5lf (%.3lf percent)",
             sum, 100.0 * (exp(sum) - 1.0) ) ;

   fclose ( fp_results ) ;
   delete cd ;

   free ( prices ) ;
   free ( inds ) ;
   free ( targets ) ;
   free ( data ) ;
   free ( lambdas ) ;
   free ( lambda_OOS ) ;
   free ( work ) ;
   exit ( 0 ) ;
}
