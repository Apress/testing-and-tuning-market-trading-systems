class SingularValueDecomp {

public:

   SingularValueDecomp ( int nrows , int ncols , int save_a=0 ) ;
   ~SingularValueDecomp () ;
   void svdcmp () ;
   void backsub ( double limit , double *soln ) ;

   int ok ;         // Was everything legal and allocs successful?

/*
   These are made public to allow access if desired.
   Normally, only 'a' (the design matrix) and 'b' (the right-hand-side)
   are written by the user.  If 'save_a' is nonzero, 'a' is kept intact.
*/

   int rows ;       // Nrows preserved here
   int cols ;       // And ncols

   double *a ;      // nrows by ncols input of design, output of U
   double *u ;      // unless save_a nonzero, in which case U output in 'u'
   double *w ;      // Unsorted ncols vector of singular values
   double *v ;      // Ncols by ncols output of 'v'
   double *b ;      // Nrows right-hand-side for backsub


private:
   void bidiag ( double *matrix ) ;
   double bid1 ( int col , double *matrix , double scale ) ;
   double bid2 ( int col , double *matrix , double scale ) ;
   void right ( double *matrix ) ;
   void left ( double *matrix ) ;
   void cancel ( int low , int high , double *matrix ) ;
   void qr ( int low , int high , double *matrix ) ;
   void qr_mrot ( int col , double sine , double cosine , double *matrix ) ;
   void qr_vrot ( int col , double sine , double cosine ) ;
   double *work ;   // Scratch vector ncols long
   double norm ;    // Norm of 'a' matrix
} ;


class StocBias {
public:
   StocBias::StocBias ( int nc ) ;
   StocBias::~StocBias () ;

   int ok ;

   void collect ( int collect_data ) ;
   void process () ;
   void compute ( double *IS_return , double *OOS_return , double *bias ) ;
   double *expose_returns () ;

private:
   int nreturns ;           // Number of returns
   int collecting ;         // Are we currently collecting data?
   int got_first_case ;     // Have we processed the first case (set of returns)?
   double *returns ;        // Returns for currently processed case
   double *IS_best ;        // In-sample best total return
   double *OOS ;            // Corresponding out-of-sample return
} ;


extern double brentmax (
   int itmax ,            // Iteration limit
   double eps ,           // Function convergence tolerance
   double tol ,           // X convergence tolerance
   double (*c_func) (double) , // Criterion function
   double *xa ,           // Lower X value, input and output
   double *xb ,           // Middle (best), input and output
   double *xc ,           // And upper, input and output
   double y               // Function value at xb
   ) ;


extern int diff_ev (
   double (*criter) ( double * , int ) , // Crit function maximized
   int nvars ,           // Number of variables
   int nints ,           // Number of first variables that are integers
   int popsize ,         // Population size
   int overinit ,        // Overinitialization for initial population
   int mintrd ,          // Minimum number of trades for candidate system
   int max_evals ,       // For safety, max number of failed initial performance evaluations; should be very large
   int max_bad_gen ,     // Max number of contiguous generations with no improvement of best
   double mutate_dev ,   // Deviation for differential mutation
   double pcross ,       // Probability of crossover
   double pclimb ,       // Probability of taking a hill-climbing step, can be zero
   double *low_bounds ,  // Lower bounds for parameters
   double *high_bounds , // And upper
   double *params ,      // Returns nvars best parameters, plus criterion at end, so must be nvars+1 long
   int print_progress ,  // Print progress to screen?
   StocBias *stoc_bias   // Optional and unrelated to differential evolution; see comments
   ) ;


extern int evec_rs ( double *mat_in , int n , int find_vec , double *vect , double *eval , double *workv ) ;

extern int glob_max (
   double low ,                // Lower limit for search
   double high ,               // Upper limit
   int npts ,                  // Number of points to try
   int log_space ,             // Space by log?
   double (*c_func) (double) , // Criterion function
   double *x1 ,
   double *y1 ,           // Lower X value and function there
   double *x2 ,
   double *y2 ,           // Middle (best)
   double *x3 ,
   double *y3             // And upper
   ) ;

extern int paramcor (
   int ncases ,     // Number of cases
   int nparams ,    // Number of parameters
   double *data     // Ncases (rows) by nparams+1 input of trial pts and f vals
   ) ;

extern void qsortds ( int first , int last , double *data , double *slave ) ;
extern void qsortdsi ( int first , int last , double *data , int *slave ) ;

int sensitivity (
   double (*criter) ( double * , int ) , // Crit function maximized
   int nvars ,           // Number of variables
   int nints ,           // Number of first variables that are integers
   int npoints ,         // Number of points at which to evaluate performance
   int nres ,            // Number of resolved points across plot
   int mintrades ,       // Minimum number of trades
   double *best ,        // Optimal parameters
   double *low_bounds ,  // Lower bounds for parameters
   double *high_bounds   // And upper
   ) ;

extern double unifrand () ;

