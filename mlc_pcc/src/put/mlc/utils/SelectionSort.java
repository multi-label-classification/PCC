package put.mlc.utils;

import java.util.Random;

/**
* This class implements the selection sort algorithm based on
* Dual-Pivot Quicksort designed by Vladimir Yaroslavskiy,
* Jon Bentley, and Josh Bloch (implementation from Java 1.7).
*
* @author Arkadiusz Jachnik
*/

public class SelectionSort {
	
	/*
     * Tuning parameters.
     */

    /**
     * maximum number of runs in merge sort
     */
    private static final int MAX_RUN_COUNT = 67;

    /**
     * maximum length of run in merge sort
     */
    private static final int MAX_RUN_LENGTH = 33;

    /**
     * if the length of an array to be sorted is less than this
     * constant, Quicksort-based algorithm is used in preference 
     * to merge sort
     */
    private static final int QUICKSORT_THRESHOLD = 286;
    
    /**
     * if the length of an array to be sorted is less than this
     * constant, insertion sort is used in preference to Quicksort
     */
    private static final int INSERTION_SORT_THRESHOLD = 47;

    /*
     * Selection algorithm with double[] array and int[] array of indexes
     */
    
	/**
     * Select k-best values from array
     *
     * @param a the array to be sorted
     * @param index the array of indexes to be sorted
     * @param kk defines the number of k best values to be selected
     */
    public static void select(double[] a, int[] index, int kk) {
        select(a, index, 0, a.length - 1, kk);
    }

    /**
     * Select k-best values from the specified range of the array
     *
     * @param a the array to be sorted
     * @param index the array of indexes to be sorted
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     * @param kk defines the number of k best values to be selected
     */
    public static void select(double[] a, int[] index, int left, int right, int kk) {
    	
    	//fill the index array
    	if(index == null || index.length != a.length) {
    		index = new int[a.length];
    		for(int i = 0; i < a.length; i++)
    			index[i] = i;
    	}
 
        //sort everything
        doSelect(a, index, left, right, kk);
    }

    /**
     * Select k-best values from the specified range of the array
     *
     * @param a the array to be sorted
     * @param index the array of indexes to be sorted
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     * @param kk defines the number of k best values to be selected
     */
    private static void doSelect(double[] a, int[] index, int left, int right, int kk) {
        // use Quicksort on small arrays
        if (right - left < QUICKSORT_THRESHOLD) {
            select(a, index, left, right, true, kk);
            return;
        }

        //index run[i] is the start of i-th run (ascending or descending sequence)
        int[] run = new int[MAX_RUN_COUNT + 1];
        int count = 0; run[0] = left;

        // check if the array is nearly sorted
        for (int k = left; k < right; run[count] = k) {
            if (a[k] < a[k + 1]) { // ascending
                while (++k <= right && a[k - 1] <= a[k]); // $codepro.audit.disable emptyStatement
            } else if (a[k] > a[k + 1]) { // descending
                while (++k <= right && a[k - 1] >= a[k]);
                for (int lo = run[count] - 1, hi = k; ++lo < --hi; ) {
                    double t = a[lo]; 
                    a[lo] = a[hi];
                    a[hi] = t;
                    
                    int indext = index[lo]; 
                    index[lo] = index[hi]; 
                    index[hi] = indext;
                }
            } else { // equal
                for (int m = MAX_RUN_LENGTH; ++k <= right && a[k - 1] == a[k]; ) {
                    if (--m == 0) {
                        select(a, index, left, right, true, kk);
                        return;
                    }
                }
            }

            //the array is not highly structured, use Quicksort instead of merge sort
            if (++count == MAX_RUN_COUNT) {
                select(a, index, left, right, true, kk);
                return;
            }
        }

        // check special cases
        if (run[count] == right++) { //the last run contains one element
            run[++count] = right;
        } else if (count == 1) { //the array is already sorted
            return;
        }
    }

    /**
     * Select k-best values from the specified range of the array by
     * Dual-Pivot-Quicksort-based algorithm
     *
     * @param a the array to be sorted
     * @param index the array of indexes to be sorted
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     * @param leftmost indicates if this part is the leftmost in the range
     * @param kk defines the number of k best values to be selected
     */
    private static void select(double[] a, int[] index, int left, int right, boolean leftmost, int kk) {
        int length = right - left + 1;
        
        //do not sort values above k
        if(left <= kk && right <= kk) {
        	return;
        }

        //use insertion sort on tiny arrays
        if (length < INSERTION_SORT_THRESHOLD) {
            if (leftmost) {
                //traditional insertion sort (without sentinel)
                for (int i = left, j = i; i < right; j = ++i) {
                    double ai = a[i + 1];
                    int indexi = index[i + 1];
                    while (ai > a[j]) {
                        a[j + 1] = a[j];
                        index[j + 1] = index[j];
                        if (j-- == left) {
                            break;
                        }
                    }
                    a[j + 1] = ai;
                    index[j + 1] = indexi;
                }
            } else {
                //skip the longest ascending sequence
                do {
                    if (left >= right) {
                        return;
                    }
                } while (a[++left] <= a[left - 1]); 

                //pair insertion sort
                for (int k = left; ++left <= right; k = ++left) {
                    double a1 = a[k], a2 = a[left];
                    int index1 = index[k], index2 = index[left];

                    if (a1 > a2) { 
                        a2 = a1; a1 = a[left];
                        index2 = index1; index1 = index[left];
                    }
                    while (a1 > a[--k]) { 
                        a[k + 2] = a[k];
                        index[k + 2] = index[k];
                    }
                    a[++k + 1] = a1;
                    index[k + 1] = index1;

                    while (a2 > a[--k]) { 
                        a[k + 1] = a[k];
                        index[k + 1] = index[k];
                    }
                    a[k + 1] = a2;
                    index[k + 1] = index2;
                }
                double last = a[right];
                int lastindex = index[right];

                while (last > a[--right]) { 
                    a[right + 1] = a[right];
                    index[right + 1] = index[right];
                }
                a[right + 1] = last;
                index[right + 1] = lastindex;
            }
            return;
        }

        //inexpensive approximation of length/7
        int seventh = (length >> 3) + (length >> 6) + 1;

        /* sort five evenly spaced elements around (and including) the
        center element in the range */
        int e3 = (left + right) >>> 1; //midpoint
        int e2 = e3 - seventh;
        int e1 = e2 - seventh;
        int e4 = e3 + seventh;
        int e5 = e4 + seventh;

        //insertion sort
        if (a[e2] > a[e1]) { 
        	double t = a[e2]; a[e2] = a[e1]; a[e1] = t; 
        	int indext = index[e2]; index[e2] = index[e1]; index[e1] = indext; 
        }

        if (a[e3] > a[e2]) { 
        	double t = a[e3]; 
        	a[e3] = a[e2]; 
        	a[e2] = t;
        	
        	int indext = index[e3]; 
        	index[e3] = index[e2]; 
        	index[e2] = indext;
        	
            if (t > a[e1]) { 
            	a[e2] = a[e1];
            	a[e1] = t; 
            	
            	index[e2] = index[e1]; 
            	index[e1] = indext;
            }
        }
        
        if (a[e4] > a[e3]) { 
        	double t = a[e4]; 
        	a[e4] = a[e3]; 
        	a[e3] = t;
        	
        	int indext = index[e4];
        	index[e4] = index[e3];
        	index[e3] = indext;
        	
            if (t > a[e2]) { 
            	a[e3] = a[e2]; 
            	a[e2] = t;
            	
            	index[e3] = index[e2]; 
            	index[e2] = indext;
            	
                if (t > a[e1]) { 
                	a[e2] = a[e1]; 
                	a[e1] = t; 
                	
                	index[e2] = index[e1]; 
                	index[e1] = indext;
                }
            }
        }
        
        if (a[e5] > a[e4]) { 
        	double t = a[e5]; 
        	a[e5] = a[e4]; 
        	a[e4] = t;
        	
        	int indext = index[e5]; 
        	index[e5] = index[e4]; 
        	index[e4] = indext;
        	
            if (t > a[e3]) { 
            	a[e4] = a[e3];
            	a[e3] = t;
            	
            	index[e4] = index[e3]; 
            	index[e3] = indext;
            	
                if (t > a[e2]) { 
                	a[e3] = a[e2]; 
                	a[e2] = t;
                	
                	index[e3] = index[e2]; 
                	index[e2] = indext;
                	
                    if (t > a[e1]) { 
                    	a[e2] = a[e1]; 
                    	a[e1] = t; 
                    	
                    	index[e2] = index[e1];
                    	index[e1] = indext;
                    }
                }
            }
        }

        int less  = left;  //the index of the first element of center part
        int great = right; //the index before the first element of right part

        if (a[e1] != a[e2] && a[e2] != a[e3] && a[e3] != a[e4] && a[e4] != a[e5]) {
            //use the second and fourth of the five sorted elements as pivots
            double pivot1 = a[e2];
            double pivot2 = a[e4];
            int indexpivot1 = index[e2];
            int indexpivot2 = index[e4];

            a[e2] = a[left];
            index[e2] = index[left];
            a[e4] = a[right];
            index[e4] = index[right];

            //Skip elements, which are less or greater than pivot values
            while (a[++less] > pivot1);
            while (a[--great] < pivot2);

            //partitioning
            outer:
            for (int k = less - 1; ++k <= great; ) {
                double ak = a[k];
                int indexk = index[k];
                if (ak > pivot1) { //move a[k] to left part
                    a[k] = a[less];
                    index[k] = index[less];
                    a[less] = ak;
                    index[less] = indexk;
                    ++less;
                } else if (ak < pivot2) { //move a[k] to right part
                    while (a[great] < pivot2) {
                        if (great-- == k) {
                            break outer;
                        }
                    }
                    if (a[great] > pivot1) { //a[great] <= pivot2 
                        a[k] = a[less];
                        index[k] = index[less];
                        a[less] = a[great];
                        index[less] = index[great];
                        ++less;
                    } else { //pivot1 <= a[great] <= pivot2
                        a[k] = a[great];
                        index[k] = index[great];
                    }
                    
                    a[great] = ak;
                    index[great] = indexk;
                    --great;
                }
            }

            //swap pivots into their final positions
            a[left]  = a[less  - 1]; a[less  - 1] = pivot1;
            index[left] = index[less - 1]; index[less - 1] = indexpivot1;
            a[right] = a[great + 1]; a[great + 1] = pivot2;
            index[right] = index[great + 1]; index[great + 1] = indexpivot2;

            //sort left and right parts recursively, excluding known pivots
            select(a, index, left, less - 2, leftmost, kk);
            if(great + 2 <= kk)
            	select(a, index, great + 2, right, false, kk);

            //if center part is too large swap internal pivot values to ends
            if (less < e1 && e5 < great) {
                //skip elements, which are equal to pivot values
                while (a[less] == pivot1) {
                    ++less;
                }

                while (a[great] == pivot2) {
                    --great;
                }

                //partitioning
                outer:
                for (int k = less - 1; ++k <= great; ) {
                    double ak = a[k];
                    int indexk = index[k];
                    if (ak == pivot1) { //move a[k] to left part
                        a[k] = a[less];
                        index[k] = index[less];
                        a[less] = ak;
                        index[less] = indexk;
                        ++less;
                    } else if (ak == pivot2) { //move a[k] to right part
                        while (a[great] == pivot2) {
                            if (great-- == k) {
                                break outer;
                            }
                        }
                        if (a[great] == pivot1) { //a[great] < pivot2
                            a[k] = a[less];
                            index[k] = index[less];
                            
                            a[less] = a[great];
                            index[less] = index[great];
                            ++less;
                        } else { //pivot1 < a[great] < pivot2
                            a[k] = a[great];
                            index[k] = index[great];
                        }
                        a[great] = ak;
                        index[great] = indexk;
                        --great;
                    }
                }
            }

            //sort center part recursively
            if(less < kk) 
            	select(a, index, less, great, false, kk);

        } else { //partitioning with one pivot
            double pivot = a[e3];
			//int indexpivot = index[e3];

            //partitioning
            for (int k = less; k <= great; ++k) {
                if (a[k] == pivot) {
                    continue;
                }
                double ak = a[k];
                int indexk = index[k];
                if (ak > pivot) { //move a[k] to left part 
                    a[k] = a[less];
                    index[k] = index[less];
                    a[less] = ak;
                    index[less] = indexk;
                    ++less;
                } else { //a[k] > pivot - move a[k] to right part
                    while (a[great] < pivot) { 
                        --great;
                    }
                    if (a[great] > pivot) { //a[great] <= pivot 
                        a[k] = a[less];
                        index[k] = index[less];
                        a[less] = a[great];
                        index[less] = index[great];
                        ++less;
                    } else { //a[great] == pivot
                        a[k] = a[great];
                        index[k] = index[great];
                    }
                    a[great] = ak;
                    index[great] = indexk;
                    --great;
                }
            }

            //sort left and right parts recursively
            select(a, index, left, less - 1, leftmost, kk);
            if(right <= kk)
            	select(a, index, great + 1, right, false, kk);
        }
    }
    
    /*
     * Selection algorithm only with int[] array of indexes
     */
    
    /**
     * Select indexes of k-best values from array
     *
     * @param a the array to be sorted
     * @param index the array of indexes to be sorted
     * @param kk defines the number of k best values to be selected
     */
    public static void selectIndexes(double[] a, int[] index, int kk) {
    	selectIndexes(a, index, 0, a.length - 1, kk);
    }

    /**
     * Select indexes of k-best values from the specified range of the array
     *
     * @param a the array to be sorted
     * @param index the array of indexes to be sorted
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     * @param kk defines the number of k best values to be selected
     */
    public static void selectIndexes(double[] a, int[] index, int left, int right, int kk) {
    	
//    	for(int i = 0; i < a.length; i++) {
//    		if(a[i] == Double.NaN) {
//    			a[i] = 0.0;
//    		}
//    	}
    	
    	//fill the index array
    	if(index == null || index.length != a.length) {
    		index = new int[a.length];
    		for(int i = 0; i < a.length; i++)
    			index[i] = i;
    	}
 
        //sort everything
        doSelectIndexes(a, index, left, right, kk);
    }

    /**
     * Select indexes of k-best values from the specified range of the array
     *
     * @param a the array to be sorted
     * @param index the array of indexes to be sorted
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     * @param kk defines the number of k best values to be selected
     */
    private static void doSelectIndexes(double[] a, int[] index, int left, int right, int kk) {
        // use Quicksort on small arrays
        if (right - left < QUICKSORT_THRESHOLD) {
            selectIndexes(a, index, left, right, true, kk);
            return;
        }

        //index run[i] is the start of i-th run (ascending or descending sequence)
        int[] run = new int[MAX_RUN_COUNT + 1];
        int count = 0; run[0] = left;

        // check if the array is nearly sorted
        for (int k = left; k < right; run[count] = k) {
            if (a[index[k]] < a[index[k + 1]]) { // ascending
                while (++k <= right && a[index[k - 1]] <= a[index[k]]);
            } else if (a[index[k]] > a[index[k + 1]]) { // descending
                while (++k <= right && a[index[k - 1]] >= a[index[k]]);
                for (int lo = run[count] - 1, hi = k; ++lo < --hi; ) { 
                    int indext = index[lo]; 
                    index[lo] = index[hi]; 
                    index[hi] = indext;
                }
            } else { // equal
                for (int m = MAX_RUN_LENGTH; ++k <= right && a[k - 1] == a[k]; ) {
                    if (--m == 0) {
                        selectIndexes(a, index, left, right, true, kk);
                        return;
                    }
                }
            }

            //the array is not highly structured, use Quicksort instead of merge sort
            if (++count == MAX_RUN_COUNT) {
                selectIndexes(a, index, left, right, true, kk);
                return;
            }
        }

        // check special cases
        if (run[count] == right++) { //the last run contains one element
            run[++count] = right;
        } else if (count == 1) { //the array is already sorted
            return;
        }
    }

    /**
     * Select indexes of k-best values from the specified range of the array by
     * Dual-Pivot-Quicksort-based algorithm
     *
     * @param a the array to be sorted
     * @param index the array of indexes to be sorted
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     * @param leftmost indicates if this part is the leftmost in the range
     * @param kk defines the number of k best values to be selected
     */
    private static void selectIndexes(double[] a, int[] index, int left, int right, boolean leftmost, int kk) {
        int length = right - left + 1;
        
        //do not sort values above k
        if(left <= kk && right <= kk) {
        	return;
        }

        //use insertion sort on tiny arrays
        if (length < INSERTION_SORT_THRESHOLD) {
            if (leftmost) {
                //traditional insertion sort (without sentinel)
                for (int i = left, j = i; i < right; j = ++i) {
                    int indexi = index[i + 1];
                    while (a[indexi] > a[index[j]]) {
                        index[j + 1] = index[j];
                        if (j-- == left) {
                            break;
                        }
                    }
                    index[j + 1] = indexi;
                }
            } else {
                //skip the longest ascending sequence
                do {
                    if (left >= right) { 
                        return;
                    }
                } while (a[index[++left]] <= a[index[left - 1]]); 

                //pair insertion sort
                for (int k = left; ++left <= right; k = ++left) {
                    int index1 = index[k], index2 = index[left];

                    if (a[index1] > a[index2]) { 
                        index2 = index1; 
                        index1 = index[left];
                    }
                    while (a[index1] > a[index[--k]]) {
                        index[k + 2] = index[k];
                    }
                    index[++k + 1] = index1;

                    while (a[index2] > a[index[--k]]) { 
                        index[k + 1] = index[k];
                    }
                    index[k + 1] = index2;
                }
                int lastindex = index[right];

                while (a[lastindex] > a[index[--right]]) { 
                    index[right + 1] = index[right];
                }
                index[right + 1] = lastindex;
            }
            return;
        }

        //inexpensive approximation of length/7
        int seventh = (length >> 3) + (length >> 6) + 1;

        /* sort five evenly spaced elements around (and including) the
        center element in the range */
        int e3 = (left + right) >>> 1; //midpoint
        int e2 = e3 - seventh;
        int e1 = e2 - seventh;
        int e4 = e3 + seventh;
        int e5 = e4 + seventh;

        //insertion sort
        if (a[index[e2]] > a[index[e1]]) { 
        	int indext = index[e2]; 
        	index[e2] = index[e1]; 
        	index[e1] = indext; 
        }

        if (a[index[e3]] > a[index[e2]]) {
        	int indext = index[e3]; 
        	index[e3] = index[e2]; 
        	index[e2] = indext;
        	
            if (a[indext] > a[index[e1]]) { 
            	index[e2] = index[e1]; 
            	index[e1] = indext;
            }
        }
        
        if (a[index[e4]] > a[index[e3]]) {
        	int indext = index[e4];
        	index[e4] = index[e3];
        	index[e3] = indext;
        	
            if (a[indext] > a[index[e2]]) { 
            	index[e3] = index[e2]; 
            	index[e2] = indext;
            	
                if (a[indext] > a[index[e1]]) { 
                	index[e2] = index[e1]; 
                	index[e1] = indext;
                }
            }
        }
        
        if (a[index[e5]] > a[index[e4]]) { 
        	int indext = index[e5]; 
        	index[e5] = index[e4]; 
        	index[e4] = indext;
        	
            if (a[indext] > a[index[e3]]) { 
            	index[e4] = index[e3]; 
            	index[e3] = indext;
            	
                if (a[indext] > a[index[e2]]) { 
                	index[e3] = index[e2]; 
                	index[e2] = indext;
                	
                    if (a[indext] > a[index[e1]]) { 
                    	index[e2] = index[e1];
                    	index[e1] = indext;
                    }
                }
            }
        }

        int less  = left;  //the index of the first element of center part
        int great = right; //the index before the first element of right part

        if (a[index[e1]] != a[index[e2]] && a[index[e2]] != a[index[e3]] && a[index[e3]] != a[index[e4]] && a[index[e4]] != a[index[e5]]) {
            //use the second and fourth of the five sorted elements as pivots
            int indexpivot1 = index[e2];
            int indexpivot2 = index[e4];

            index[e2] = index[left];
            index[e4] = index[right];

            //Skip elements, which are less or greater than pivot values
            while (a[index[++less]] > a[indexpivot1]);
            while (a[index[--great]] < a[indexpivot2]);

            //partitioning
            outer:
            for (int k = less - 1; ++k <= great; ) {
                int indexk = index[k];
                if (a[indexk] > a[indexpivot1]) { //move a[k] to left part
                    index[k] = index[less];
                    index[less] = indexk;
                    ++less;
                } else if (a[indexk] < a[indexpivot2]) { //move a[k] to right part
                    while (a[index[great]] < a[indexpivot2]) {
                        if (great-- == k) {
                            break outer;
                        }
                    }
                    if (a[index[great]] > a[indexpivot1]) { //a[great] <= pivot2 
                        index[k] = index[less];
                        index[less] = index[great];
                        ++less;
                    } else { //pivot1 <= a[great] <= pivot2
                        index[k] = index[great];
                    }
                    
                    index[great] = indexk;
                    --great;
                }
            }

            //swap pivots into their final positions
            index[left] = index[less - 1]; index[less - 1] = indexpivot1;
            index[right] = index[great + 1]; index[great + 1] = indexpivot2;

            //sort left and right parts recursively, excluding known pivots
            selectIndexes(a, index, left, less - 2, leftmost, kk);
            if(great + 2 <= kk)
            	selectIndexes(a, index, great + 2, right, false, kk);

            //if center part is too large swap internal pivot values to ends
            if (less < e1 && e5 < great) {
                //skip elements, which are equal to pivot values
                while (a[index[less]] == a[indexpivot1]) {
                    ++less;
                }

                while (a[index[great]] == a[indexpivot2]) {
                    --great;
                }

                //partitioning
                outer:
                for (int k = less - 1; ++k <= great; ) {
                    int indexk = index[k];
                    if (a[indexk] == a[indexpivot2]) { //move a[k] to left part
                        index[k] = index[less];
                        index[less] = indexk;
                        ++less;
                    } else if (a[indexk] == a[indexpivot1]) { //move a[k] to right part
                        while (a[index[great]] == a[indexpivot1]) {
                            if (great-- == k) {
                                break outer;
                            }
                        }
                        if (a[index[great]] == a[indexpivot2]) { //a[great] < pivot2
                            index[k] = index[less];
                            index[less] = index[great];
                            ++less;
                        } else { //pivot1 < a[great] < pivot2
                            index[k] = index[great];
                        }
                        index[great] = indexk;
                        --great;
                    }
                }
            }

            //sort center part recursively
            if(less <= kk) 
            	selectIndexes(a, index, less, great, false, kk);

        } else { //partitioning with one pivot
            int indexpivot = index[e3];

            //partitioning
            for (int k = less; k <= great; ++k) {
                if (a[index[k]] == a[indexpivot]) {
                    continue;
                }
                int indexk = index[k];
                if (a[indexk] > a[indexpivot]) { //move a[k] to left part 
                    index[k] = index[less];
                    index[less] = indexk;
                    ++less;
                } else { //a[k] > pivot - move a[k] to right part
                    while (a[index[great]] < a[indexpivot]) { 
                        --great;
                    }
                    if (a[index[great]] > a[indexpivot]) { //a[great] <= pivot 
                        index[k] = index[less];
                        index[less] = index[great];
                        ++less;
                    } else { //a[great] == pivot
                        index[k] = index[great];
                    }
                    index[great] = indexk;
                    --great;
                }
            }

            //sort left and right parts recursively
            selectIndexes(a, index, left, less - 1, leftmost, kk);
            if(left <= kk)
            	selectIndexes(a, index, great + 1, right, false, kk);
        }
    }
    
    /*
     * Old versions of selection algorithms
     */
    
    /**
     * Partition algorithm
     *
     * @param input the array to be sorted
     * @param index the array of indexes to be sorted
     * @param p the index of the first element
     * @param q the index of the last element
     */
    public static int partition(double[] input, int[] index, int p, int q) {
		
		double x = input[p];
		int i = p;
		for (int j = (p + 1); j <= q; j++) {
			if (input[j] >= x) { 
				i = i + 1;
				if(i < j){
					double t1 = input[j];
					input[j] = input[i];
					input[i] = t1;
					
					int t2 = index[j];
					index[j] = index[i];
					index[i] = t2;
					
				}
			}
		}
		double t11 = input[p];
		input[p] = input[i];
		input[i] = t11;
		
		int t21 = index[p];
		index[p] = index[i];
		index[i] = t21;
		
		
		return i;
	}
		
    /**
     * Select k-best values from the specified range of the array (old version)
     *
     * @param list the array to be sorted
     * @param index the array of indexes to be sorted
     * @param left the index of the first element, inclusive, to be sorted
     * @param right the index of the last element, inclusive, to be sorted
     * @param k defines the number of k best values to be selected
     */
	public static void select2(double[] list, int[] index, int left, int right, int k){
		int pivotIndex = partition(list, index, left, right);
		if(pivotIndex == k){
			return;
		}
		else if(k < pivotIndex) {
			select2(list, index, left, pivotIndex -1, k);
		} else {
			select2(list, index, pivotIndex +1 , right, k);
		}
	
	}
	
	public static int rand(int min, int max) {
		Random r = new Random();
		return r.nextInt(max - min + 1) + min;
	}

	

}
