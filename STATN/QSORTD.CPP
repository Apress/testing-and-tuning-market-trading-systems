/******************************************************************************/
/*                                                                            */
/*  QSORT - Quick sort a double array.                                        */
/*                                                                            */
/******************************************************************************/

void qsortd ( int first , int last , double *data )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortd ( first , upper , data ) ;
   if (lower < last)
      qsortd ( lower , last , data ) ;
}

void qsortds ( int first , int last , double *data , double *slave )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds ( first , upper , data , slave ) ;
   if (lower < last)
      qsortds ( lower , last , data , slave ) ;
}

void qsortds2 ( int first , int last , double *data , double *slave , double *slave2 )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         ftemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = ftemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds2 ( first , upper , data , slave , slave2 ) ;
   if (lower < last)
      qsortds2 ( lower , last , data , slave , slave2 ) ;
}

void qsortds3 ( int first , int last , double *data , double *slave ,
                double *slave2 , double *slave3 )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         ftemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = ftemp ;
         ftemp = slave3[lower] ;
         slave3[lower] = slave3[upper] ;
         slave3[upper] = ftemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds3 ( first , upper , data , slave , slave2 , slave3 ) ;
   if (lower < last)
      qsortds3 ( lower , last , data , slave , slave2 , slave3 ) ;
}

void qsortds4 ( int first , int last , double *data , double *slave ,
                double *slave2 , double *slave3 , double *slave4 )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         ftemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = ftemp ;
         ftemp = slave3[lower] ;
         slave3[lower] = slave3[upper] ;
         slave3[upper] = ftemp ;
         ftemp = slave4[lower] ;
         slave4[lower] = slave4[upper] ;
         slave4[upper] = ftemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds4 ( first , upper , data , slave , slave2 , slave3 , slave4 ) ;
   if (lower < last)
      qsortds4 ( lower , last , data , slave , slave2 , slave3 , slave4 ) ;
}

void qsortds5 ( int first , int last , double *data , double *slave ,
                double *slave2 , double *slave3 , double *slave4 ,
                double *slave5 )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         ftemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = ftemp ;
         ftemp = slave3[lower] ;
         slave3[lower] = slave3[upper] ;
         slave3[upper] = ftemp ;
         ftemp = slave4[lower] ;
         slave4[lower] = slave4[upper] ;
         slave4[upper] = ftemp ;
         ftemp = slave5[lower] ;
         slave5[lower] = slave5[upper] ;
         slave5[upper] = ftemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds5 ( first , upper , data , slave , slave2 , slave3 , slave4 , slave5 ) ;
   if (lower < last)
      qsortds5 ( lower , last , data , slave , slave2 , slave3 , slave4 , slave5 ) ;
}

void qsortds6 ( int first , int last , double *data , double *slave ,
                double *slave2 , double *slave3 , double *slave4 ,
                double *slave5 , double *slave6 )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         ftemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = ftemp ;
         ftemp = slave3[lower] ;
         slave3[lower] = slave3[upper] ;
         slave3[upper] = ftemp ;
         ftemp = slave4[lower] ;
         slave4[lower] = slave4[upper] ;
         slave4[upper] = ftemp ;
         ftemp = slave5[lower] ;
         slave5[lower] = slave5[upper] ;
         slave5[upper] = ftemp ;
         ftemp = slave6[lower] ;
         slave6[lower] = slave6[upper] ;
         slave6[upper] = ftemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds6 ( first , upper , data , slave , slave2 , slave3 , slave4 , slave5 , slave6 ) ;
   if (lower < last)
      qsortds6 ( lower , last , data , slave , slave2 , slave3 , slave4 , slave5 , slave6 ) ;
}

void qsortds7 ( int first , int last , double *data , double *slave ,
                double *slave2 , double *slave3 , double *slave4 ,
                double *slave5 , double *slave6 , double *slave7 )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         ftemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = ftemp ;
         ftemp = slave3[lower] ;
         slave3[lower] = slave3[upper] ;
         slave3[upper] = ftemp ;
         ftemp = slave4[lower] ;
         slave4[lower] = slave4[upper] ;
         slave4[upper] = ftemp ;
         ftemp = slave5[lower] ;
         slave5[lower] = slave5[upper] ;
         slave5[upper] = ftemp ;
         ftemp = slave6[lower] ;
         slave6[lower] = slave6[upper] ;
         slave6[upper] = ftemp ;
         ftemp = slave7[lower] ;
         slave7[lower] = slave7[upper] ;
         slave7[upper] = ftemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds7 ( first , upper , data , slave , slave2 , slave3 , slave4 , slave5 , slave6 , slave7 ) ;
   if (lower < last)
      qsortds7 ( lower , last , data , slave , slave2 , slave3 , slave4 , slave5 , slave6 , slave7 ) ;
}

void qsortds8 ( int first , int last , double *data , double *slave ,
                double *slave2 , double *slave3 , double *slave4 ,
                double *slave5 , double *slave6 , double *slave7 ,
                double *slave8 )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         ftemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = ftemp ;
         ftemp = slave3[lower] ;
         slave3[lower] = slave3[upper] ;
         slave3[upper] = ftemp ;
         ftemp = slave4[lower] ;
         slave4[lower] = slave4[upper] ;
         slave4[upper] = ftemp ;
         ftemp = slave5[lower] ;
         slave5[lower] = slave5[upper] ;
         slave5[upper] = ftemp ;
         ftemp = slave6[lower] ;
         slave6[lower] = slave6[upper] ;
         slave6[upper] = ftemp ;
         ftemp = slave7[lower] ;
         slave7[lower] = slave7[upper] ;
         slave7[upper] = ftemp ;
         ftemp = slave8[lower] ;
         slave8[lower] = slave8[upper] ;
         slave8[upper] = ftemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds8 ( first , upper , data , slave , slave2 , slave3 , slave4 ,
                 slave5 , slave6 , slave7 , slave8 ) ;
   if (lower < last)
      qsortds8 ( lower , last , data , slave , slave2 , slave3 , slave4 ,
                 slave5 , slave6 , slave7 , slave8 ) ;
}

void qsortds9 ( int first , int last , double *data , double *slave ,
                double *slave2 , double *slave3 , double *slave4 ,
                double *slave5 , double *slave6 , double *slave7 ,
                double *slave8 , double *slave9 )
{
   int lower, upper ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         ftemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = ftemp ;
         ftemp = slave3[lower] ;
         slave3[lower] = slave3[upper] ;
         slave3[upper] = ftemp ;
         ftemp = slave4[lower] ;
         slave4[lower] = slave4[upper] ;
         slave4[upper] = ftemp ;
         ftemp = slave5[lower] ;
         slave5[lower] = slave5[upper] ;
         slave5[upper] = ftemp ;
         ftemp = slave6[lower] ;
         slave6[lower] = slave6[upper] ;
         slave6[upper] = ftemp ;
         ftemp = slave7[lower] ;
         slave7[lower] = slave7[upper] ;
         slave7[upper] = ftemp ;
         ftemp = slave8[lower] ;
         slave8[lower] = slave8[upper] ;
         slave8[upper] = ftemp ;
         ftemp = slave9[lower] ;
         slave9[lower] = slave9[upper] ;
         slave9[upper] = ftemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds9 ( first , upper , data , slave , slave2 , slave3 , slave4 ,
                 slave5 , slave6 , slave7 , slave8 , slave9 ) ;
   if (lower < last)
      qsortds9 ( lower , last , data , slave , slave2 , slave3 , slave4 ,
                 slave5 , slave6 , slave7 , slave8 , slave9 ) ;
}

void qsortdsi ( int first , int last , double *data , int *slave )
{
   int lower, upper, itemp ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         itemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = itemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortdsi ( first , upper , data , slave ) ;
   if (lower < last)
      qsortdsi ( lower , last , data , slave ) ;
}

void qsortds64 ( int first , int last , double *data , unsigned _int64 *slave )
{
   int lower, upper ;
   unsigned _int64 itemp ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         itemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = itemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortds64 ( first , upper , data , slave ) ;
   if (lower < last)
      qsortds64 ( lower , last , data , slave ) ;
}

void qsortdsri ( int first , int last , double *data , double *slave , int *slave2 )
{
   int lower, upper, itemp ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = ftemp ;
         itemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = itemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortdsri ( first , upper , data , slave , slave2 ) ;
   if (lower < last)
      qsortdsri ( lower , last , data , slave , slave2 ) ;
}

void qsortdsii ( int first , int last , double *data , int *slave , int *slave2 )
{
   int lower, upper, itemp ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         itemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = itemp ;
         itemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = itemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortdsii ( first , upper , data , slave , slave2 ) ;
   if (lower < last)
      qsortdsii ( lower , last , data , slave , slave2 ) ;
}

void qsorti ( int first , int last , int *data )
{
   int lower, upper ;
   int ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsorti ( first , upper , data ) ;
   if (lower < last)
      qsorti ( lower , last , data ) ;
}

void qsortisi ( int first , int last , int *data , int *slave )
{
   int lower, upper, itemp ;
   int ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         itemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = itemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortisi ( first , upper , data , slave ) ;
   if (lower < last)
      qsortisi ( lower , last , data , slave ) ;
}

void qsortisd ( int first , int last , int *data , double *slave )
{
   int lower, upper ;
   int ftemp, split ;
   double dtemp ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         dtemp = slave[lower] ;
         slave[lower] = slave[upper] ;
         slave[upper] = dtemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortisd ( first , upper , data , slave ) ;
   if (lower < last)
      qsortisd ( lower , last , data , slave ) ;
}

void qsortissii ( int first , int last , int *data , short int *slave1 , int *slave2 )
{
   int lower, upper, itemp ;
   int ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         itemp = slave1[lower] ;
         slave1[lower] = slave1[upper] ;
         slave1[upper] = (short int) itemp ;
         itemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = itemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortissii ( first , upper , data , slave1 , slave2 ) ;
   if (lower < last)
      qsortissii ( lower , last , data , slave1 , slave2 ) ;
}


void qsort64ssii ( int first , int last , unsigned _int64 *data , short int *slave1 , int *slave2 )
{
   int lower, upper, itemp ;
   unsigned _int64 ftemp, split ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         itemp = slave1[lower] ;
         slave1[lower] = slave1[upper] ;
         slave1[upper] = (short int) itemp ;
         itemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = itemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsort64ssii ( first , upper , data , slave1 , slave2 ) ;
   if (lower < last)
      qsort64ssii ( lower , last , data , slave1 , slave2 ) ;
}


void qsortid4 ( int first , int last , int *data , double *slave1 , double *slave2 , double *slave3 , double *slave4 )
{
   int lower, upper ;
   int ftemp, split ;
   double dtemp ;

   split = data[(first+last)/2] ;
   lower = first ;
   upper = last ;

   do {
      while ( split > data[lower] )
         ++lower ;
      while ( split < data[upper] )
         --upper ;
      if (lower == upper) {
         ++lower ;
         --upper ;
         }
      else if (lower < upper) {
         dtemp = slave1[lower] ;
         slave1[lower] = slave1[upper] ;
         slave1[upper] = dtemp ;
         dtemp = slave2[lower] ;
         slave2[lower] = slave2[upper] ;
         slave2[upper] = dtemp ;
         dtemp = slave3[lower] ;
         slave3[lower] = slave3[upper] ;
         slave3[upper] = dtemp ;
         dtemp = slave4[lower] ;
         slave4[lower] = slave4[upper] ;
         slave4[upper] = dtemp ;
         ftemp = data[lower] ;
         data[lower++] = data[upper] ;
         data[upper--] = ftemp ;
         }
      } while ( lower <= upper ) ;

   if (first < upper)
      qsortid4 ( first , upper , data , slave1 , slave2 , slave3 , slave4 ) ;
   if (lower < last)
      qsortid4 ( lower , last , data , slave1 , slave2 , slave3 , slave4 ) ;
}
