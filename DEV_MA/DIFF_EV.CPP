/******************************************************************************/
/*                                                                            */
/*  DIFF_EV - Differential evolution optimization                             */
/*                                                                            */
/*  Popsize should be 5 to 10 times n, more for a more global search.         */
/*  Overinit should be 0 for simple problems, or popsize for hard problems.   */
/*  Mutate_dev should be about 0.4 to 1.2, with larger values giving a more   */
/*  global search.                                                            */
/*  Pcross should be 0-1.  This is the probability that each parameter in     */
/*  crossover will be chosen from the noisy parent (as opposed to the pure    */
/*  parent against which the child will be tested).                           */
/*  The authors state that small values (like 0.1) produce a more global      */
/*  solution, but that is opposite my intuition.                              */
/*                                                                            */
/******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "headers.h"

// These are for passing needed information to the maximiztion routines

static int local_ivar ;      // Which variable within vartype
static double local_base, *local_x ;
static int local_nvars , local_nints ;
static int local_mintrades ;          // This will be reduced if multiple failures
static double *local_low_bounds , *local_high_bounds ;
static double (*local_criter) ( double *params , int mintrades ) ;

static double c_func ( double param ) ;

double ensure_legal ( int nvars , int nints , double *low_bounds , double *high_bounds , double *params ) ;

int diff_ev (
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
   )
{
   int i, j, k, ind, ivar, ibase, dim, n_evals, bad_generations ;
   int ilow, ihigh, ret_code, generation, ibest, n_tweaked, improved ;
   int used_mutated_parameter, failures, success ;
   double *pop1, *pop2, *best, *popptr, value, worstf, avgf, avg;
   double dtemp, *old_gen, *new_gen, *minptr, test_val, old_value ;
   double *parent1, *parent2, grand_best, *dest_ptr, *diff1, *diff2 ;
   double x1, y1, x2, y2, x3, y3, lower, upper ;

   ret_code = 0 ;

   local_mintrades = mintrd ;

/*
   Allocate memory to store the two populations (old and new).
*/

   dim = nvars + 1 ;  // Each case is nvars variables plus criterion

   pop1 = (double *) malloc ( dim * popsize * sizeof(double)) ;
   pop2 = (double *) malloc ( dim * popsize * sizeof(double)) ;
   best = (double *) malloc ( dim * sizeof(double)) ;

   if (pop1 == NULL  ||  pop2 == NULL  ||  best == NULL) {
      if (pop1 != NULL)
         free ( pop1 ) ;
      if (pop2 != NULL)
         free ( pop2 ) ;
      if (best != NULL)
         free ( best ) ;
      return 1 ;  // Error flag
      }

/*
-------------------------------------------------------------------------
   Generate the initial population.
   Once we reach the popsize, if we are going into overinit then each new
   case replaces the worst so far if it is better.
   Temporarily store this trial case in the first slot of pop2.
-------------------------------------------------------------------------
*/

   failures = 0 ;                         // Counts consecutive failures
   n_evals = 0 ;                          // Counts evaluations for catastrophe escape
   stoc_bias->collect ( 1 ) ;             // Turn on StocBias data collection
                                          // This is unrelated to differential evolution and may be omitted

   for (ind=0 ; ind<popsize+overinit ; ind++) {

      if (ind < popsize)                  // If we are in pop1
         popptr = pop1 + ind * dim ;      // Point to the slot in pop1
      else                                // Use first slot in pop2 for work
         popptr = pop2 ;                  // Point to first slot in pop2

      for (i=0 ; i<nvars ; i++) {         // For all variables

         if (i < nints) {                 // Is this an integer?
            popptr[i] = low_bounds[i] + (int) (unifrand () * (high_bounds[i] - low_bounds[i] + 1.0)) ;
            if (popptr[i] > high_bounds[i])  // Virtually impossible, but be safe
               popptr[i] = high_bounds[i] ;
            }

         else                             // real
            popptr[i] = low_bounds[i] + (unifrand () * (high_bounds[i] - low_bounds[i])) ;
         } // For all parameters

      value = criter ( popptr , local_mintrades ) ;
      popptr[nvars] = value ;          // Also save criterion after variables
      ++n_evals ;                      // Count evaluations for emergency escape

      if (ind == 0) {
         grand_best = worstf = avgf = value ;
         memcpy ( best , pop1 , dim * sizeof(double) ) ; // Best so far is first!
         }

      if (value <= 0.0) {  // If this individual is totally worthless
         if (n_evals > max_evals)  // Safety escape should ideally never happen
            goto FINISHED ;
         --ind ;           // Skip it entirely
         if (++failures >= 500) {  // This many in a row
            failures = 0 ;
            local_mintrades = local_mintrades * 9 / 10 ;
            if (local_mintrades < 1)
               local_mintrades = 1 ;
            }
         continue ;
         }
      else
         failures = 0 ;

/*
   Maintain best, worst, and average
   These are strictly for user updates, and they have nothing to do with the algorithm.
   Well, we do keep the grand best throughout, as this is what is ultimately returned.
*/

      if (value > grand_best) {   // Best ever
         memcpy ( best , popptr , dim * sizeof(double) ) ;
         grand_best = value ;
         }

      if (value < worstf)
         worstf = value ;

      avgf += value ;

      if (print_progress) {
         if (ind < popsize)        // Before overinit we update average as each new trial is done
            avg = avgf / (ind+1) ;
         else                      // After overinit we examine only the population
            avg = avgf / popsize ;
         printf ( "\n%d: Val=%.4lf Best=%.4lf Worst=%.4lf Avg=%.4lf  (fail rate=%.1lf)", ind, value, grand_best, worstf, avg, n_evals / (ind+1.0) ) ;
         for (i=0 ; i<nvars ; i++)
            printf ( " %.4lf", popptr[i] ) ;
         }

/*
   If we have finished pop1 and we are now into overinit, the latest
   candidate is in the first slot of pop2.  Search pop1 for the worst
   individual.  If the new candidate is better than the worst in pop1,
   replace the old with the new.
   We recompute the average within the original population.
*/

      if (ind >= popsize) {      // If we finished pop1, now doing overinit
         avgf = 0.0 ;
         minptr = NULL ;  // Not needed.  Shuts up 'use before define'
         for (i=0 ; i<popsize ; i++) {  // Search pop1 for the worst
            dtemp = (pop1+i*dim)[nvars] ;
            avgf += dtemp ;
            if ((i == 0)  ||  (dtemp < worstf)) {
               minptr = pop1 + i * dim ;
               worstf = dtemp ;
               }
            } // Searching pop1 for worst
         if (value > worstf) {  // If this is better than the worst, replace worst with it
            memcpy ( minptr , popptr , dim * sizeof(double) ) ;
            avgf += value - worstf ;  // Account for the substitution
            }
         } // If doing overinit

      } // For all individuals (population and overinit)

   stoc_bias->collect ( 0 ) ;             // Turn off StocBias data collection
                                          // This is unrelated to differential evolution and may be omitted

/*
   We now have the initial population and also have completed overinitialization.
   Search the initial population to find the subscript of the best.
   This is to let us periodically tweak the best.
*/

   ibest = n_tweaked = 0 ;
   value = pop1[nvars] ;
   for (ind=1 ; ind<popsize ; ind++) {
      popptr = pop1 + ind * dim ;      // Point to the slot in pop1
      if (popptr[nvars] > value) {
         value = popptr[nvars] ;
         ibest = ind ;
         }
      }

/*
--------------------------------------------------------------------------------

   This is the main loop.  For each generation, use old_gen for the parents
   and create the children in new_gen.  These flip between pop1 and pop2.
   'Repeats' counts the number of generations with no improvements.
   This allows automatic escape for batch runs.

--------------------------------------------------------------------------------
*/

   old_gen = pop1 ;       // This is the old, parent generation
   new_gen = pop2 ;       // The children will be produced here
   bad_generations = 0 ;  // Counts contiguous generations with no improvement of best

   for (generation=1 ; ; generation++) {

      worstf = 1.e60 ;
      avgf = 0.0 ;
      improved = 0 ;      // Will flag if we improved in this generation

      for (ind=0 ; ind<popsize ; ind++) {    // Generate all children

         parent1 = old_gen + ind * dim ;     // Pure (and tested) parent
         dest_ptr = new_gen + ind * dim ;    // Winner goes here for next gen

/*
   Generate three different random numbers for parent2 and the differentials
*/

         do { i = (int) (unifrand() * popsize) ; }
            while ( i >= popsize || i == ind ) ;

         do { j = (int) (unifrand() * popsize) ; }
            while ( j >= popsize || j == ind || j == i ) ;

         do { k = (int) (unifrand() * popsize) ; }
            while ( k >= popsize || k == ind || k == i || k == j ) ;

         parent2 = old_gen + i * dim ;      // Parent to mutate
         diff1 = old_gen + j * dim ;        // First differential vector
         diff2 = old_gen + k * dim ;        // Second differential vector

/*
   Build the child in the destination array, even though it may have to be
   overwritten with parent1 if this child is not superior.

   We need to randomly pick a starting point in the parameter vector because
   when we get to the end if we have not used any parameters from the mutation
   vector we force one into the child.  We do not want that forced parameter
   to always be the last position in the vector!
*/

         do { j = (int) (unifrand() * nvars) ; }
            while ( j >= nvars ) ;  // Pick a starting parameter

         used_mutated_parameter = 0 ;

         for (i=nvars-1 ; i>=0 ; i--) {
            if ((i == 0 && ! used_mutated_parameter) || (unifrand() < pcross)) {
               dest_ptr[j] = parent2[j] + mutate_dev * (diff1[j] - diff2[j]) ;
               used_mutated_parameter = 1 ;
               }   // We mutated this variable
            else   // We did not mutate this variable, so copy old value
               dest_ptr[j] = parent1[j] ;
            j = (j + 1) % nvars ;   // Rotate through all variables
            }

/*
   For all parameters, the above operation may have pushed the value outside
   its legal limit.  For integer parameters, illegal values have
   almost certainly resulted.  Fix these problems.
*/

         ensure_legal ( nvars , nints , low_bounds , high_bounds , dest_ptr ) ;
         
/*
   Mutation is complete.  Evaluate the performance of this child.
   If the child is better than parent1, keep it right here in the destination
   array where it was created.  (Put its criterion there too.)
   If it is inferior to parent1, move that parent and its criterion to the
   destination array.
*/

         value = criter ( dest_ptr , local_mintrades ) ;

         if (value > parent1[nvars]) {  // If the child is better than parent1
            dest_ptr[nvars] = value ;   // Get the child's value (The vars are already there)
            if (value > grand_best) {   // And update best so far
               grand_best = value ;
               memcpy ( best , dest_ptr , dim * sizeof(double) ) ;
               ibest = ind ;
               n_tweaked = 0 ;
               improved = 1 ;   // Flag that the best improved in this generation
               }
            }

         else {                         // Else copy parent1 and its value
            memcpy ( dest_ptr , parent1 , dim * sizeof(double) ) ;
            value = parent1[nvars] ;
            }

/*
   If we are to randomly tweak (do a hill-climbing step), do it now.
   Note that it is rarely possible for this step to cause the 'worst' to
   get worse!  The glob_max routine may search an interval that does not
   have the current parameters as a trial point, and never find anything
   quite as good.  This should happen only very late in the game, and not
   have any bad consequences.

   We use n_tweaked to count how many times this particular 'grand best'
   has been tweaked.  It is incremented each time the grand best is tweaked,
   and it is reset to zero whenever we get a new grand best.

   In order to do this hill-climbing step, we must have:
      pclimb > 0.0 (The user is allowing hill climbing) AND
         (This individual is the grand best AND we have not yet tweaked every variable) OR
         We randomly tweak some variable in this individual
*/

         if (pclimb > 0.0  &&
                     ((ind == ibest  &&  n_tweaked < nvars)  ||  (unifrand() < pclimb))) {
            if (ind == ibest) {          // Once each generation tweak the best
               ++n_tweaked ;             // But quit if done all vars
               k = generation % nvars ;  // Cycle through all vars
               }
            else {                       // Randomly choose an individual
               k = (int) (unifrand() * nvars) ; // Which var to optimize
               if (k >= nvars)           // Safety only
                  k = nvars - 1 ;
               }

            // Handle integer parameters

            if (k < nints) {             // Is this an integer?
               ivar = ibase = (int) dest_ptr[k] ;
               ilow = (int) low_bounds[k] ;
               ihigh = (int) high_bounds[k] ;
               success = 0 ;
               if (print_progress)
                  printf ( "\nCriterion maximization of individual %d integer variable %d from %d = %.6lf", ind, k, ibase, value ) ;
               while (++ivar <= ihigh) {
                  dest_ptr[k] = ivar ;
                  test_val = criter ( dest_ptr , local_mintrades ) ;
                  if (print_progress)
                     printf ( "\n  %d = %.6lf", ivar, test_val ) ;
                  if (test_val > value) {
                     value = test_val ;
                     ibase = ivar ;
                     success = 1 ;
                     }
                  else {
                     dest_ptr[k] = ibase ;
                     break ;
                     }
                  }

               if (! success) {
                  ivar = ibase ;
                  while (--ivar >= ilow) {
                     dest_ptr[k] = ivar ;
                     test_val = criter ( dest_ptr , local_mintrades ) ;
                     if (print_progress)
                        printf ( "\n  %d = %.6lf", ivar, test_val ) ;
                     if (test_val > value) {
                        value = test_val ;
                        ibase = ivar ;
                        success = 1 ;
                        }
                     else {
                        dest_ptr[k] = ibase ;
                        break ;
                        }
                     }
                  }
               if (print_progress) {
                  if (success)
                     printf ( "\nSuccess at %.0lf = %.6lf", dest_ptr[k], value ) ;
                  else
                     printf ( "\nNo success at %.0lf = %.6lf", dest_ptr[k], value ) ;
                  }
               } // If k < nints (this parameter is an integer)

            // Handle real parameters

            else {                          // This is a real parameter
               local_criter = criter ;
               local_ivar = k ;             // Pass it to criterion routine
               local_base = dest_ptr[k] ;   // Preserve orig var
               local_x = dest_ptr ;
               local_nvars = nvars ;
               local_nints = nints ;
               local_low_bounds = low_bounds ;
               local_high_bounds = high_bounds ;
               old_value = value ;
               if (print_progress)
                  printf ( "\nCriterion maximization of individual %d real variable %d from %.5lf = %.6lf", ind, k, local_base, value ) ;
               lower = local_base - 0.1 * (high_bounds[k] - low_bounds[k]) ;
               upper = local_base + 0.1 * (high_bounds[k] - low_bounds[k]) ;
               if (lower < low_bounds[k]) {
                  lower = low_bounds[k] ;
                  upper = low_bounds[k] + 0.2 * (high_bounds[k] - low_bounds[k]) ;
                  }
               if (upper > high_bounds[k]) {
                  upper = high_bounds[k] ;
                  lower = high_bounds[k] - 0.2 * (high_bounds[k] - low_bounds[k]) ;
                  }
               k = glob_max ( lower , upper , 7 , 0 , c_func ,
                              &x1 , &y1 , &x2 , &y2 , &x3 , &y3 ) ;
               brentmax ( 5 , 1.e-8 , 0.0001 , c_func , &x1 , &x2 , &x3 , y2 ) ;
               dest_ptr[local_ivar] = x2 ;  // Optimized var value
               ensure_legal ( nvars , nints , low_bounds , high_bounds , dest_ptr ) ;
               value = criter ( dest_ptr , local_mintrades ) ;
               if (value > old_value) {
                  dest_ptr[nvars] = value ;
                  if (print_progress)
                     printf ( "\nSuccess at %.5lf = %.6lf", dest_ptr[local_ivar], value ) ;
                  }
               else {
                  dest_ptr[local_ivar] = local_base ;    // Restore original value
                  value = old_value ;
                  if (print_progress)
                     printf ( "\nNo success at %.5lf = %.6lf", dest_ptr[local_ivar], value ) ;
                  }
               if (value > grand_best) {      // Update best so far
                  grand_best = value ;
                  memcpy ( best , dest_ptr , dim * sizeof(double) ) ;
                  ibest = ind ;
                  n_tweaked = 0 ;
                  improved = 1 ;   // Flag that the best improved in this generation
                  }
               } // If optimizing real parameter
            } // If doing hill-climbing step

         if (value < worstf)
            worstf = value ;

         avgf += value ;

         } // Create all children in this generation

      if (print_progress) {
         printf ( "\nGen %d Best=%.4lf Worst=%.4lf Avg=%.4lf", generation, grand_best, worstf, avgf/popsize ) ;
         for (i=0 ; i<nvars ; i++)
            printf ( " %.4lf", best[i] ) ;
         }

/*
   This generation is complete.  See if we must quit due to too many contiguous failures to improve.
   Reverse old_gen and new_gen in pop1 and pop2.
*/

      if (! improved) {
         ++bad_generations ;
         if (bad_generations > max_bad_gen)
            break ;
         }
      else
         bad_generations = 0 ;

      if (old_gen == pop1) {
         old_gen = pop2 ;
         new_gen = pop1 ;
         }
      else {
         old_gen = pop1 ;
         new_gen = pop2 ;
         }

      } // For all generations

/*
   Compute and print parameter correlations et cetera
*/

   if (paramcor ( popsize , nvars , new_gen ))
      ret_code = -1 ;

/*
   We are done.
*/

FINISHED:

   memcpy ( params , best , dim * sizeof(double) ) ;

   free ( pop1 ) ;
   free ( pop2 ) ;
   free ( best ) ;

   return ret_code ;
}


/*
--------------------------------------------------------------------------------

   Local function ensures that all parameters are legal.
   This also returns a penalty for illegal parameters.
   The penalty is only used for tweaking (glob_max can extend the search
   forever!).

--------------------------------------------------------------------------------
*/

static double ensure_legal ( int nvars , int nints , double *low_bounds , double *high_bounds , double *params )
{
   int i ;
   double penalty ;

   penalty = 0.0 ;

   for (i=0 ; i<nvars ; i++) {
      if (i < nints) {
         if (params[i] >= 0)
            params[i] = (int) (params[i] + 0.5) ;
         else if (params[i] < 0)
            params[i] = -(int) (0.5 - params[i]) ;
         }
      if (params[i] > high_bounds[i]) {
         penalty += 1.e10 * (params[i] - high_bounds[i]) ;
         params[i] = high_bounds[i] ;
         }
      if (params[i] < low_bounds[i]) {
         penalty += 1.e10 * (low_bounds[i] - params[i]) ;
         params[i] = low_bounds[i] ;
         }
      }

   return penalty ;
}

/*
--------------------------------------------------------------------------------

   Local function for univariate optimization

--------------------------------------------------------------------------------
*/

static double c_func ( double param )
{
   double penalty ;

   local_x[local_ivar] = param ;
   penalty = ensure_legal ( local_nvars , local_nints , local_low_bounds , local_high_bounds , local_x ) ;
   return local_criter ( local_x , local_mintrades ) - penalty ;
}