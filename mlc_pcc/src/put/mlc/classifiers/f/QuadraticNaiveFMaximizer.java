/**
 * 
 */
package put.mlc.classifiers.f;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;

/**
 * Implementation of the cubic time and quadratic time algorithms for computing
 * the predictions maximizing the expected F-measure under the label independence
 * assumption. The algorithms are described in <i>Nan Ye, Adam K.M. Chai, Wee
 * Sun Lee, Hai Leong Chieu. Optimizing F-measures: A Tale of Two Approaches.
 * ICML 2012.</i>
 * 
 * In this implementation, <i>F<sub>beta</sub> = (1+beta)/(1/prec+beta/rec)
 * </i>, where prec and rec are the precision and recall respectively.
 * 
 * This code is strongly based on the implementation of Ye Nan (<a
 * href="mailto:yenan@comp.nus.edu.sg">yenan@comp.nus.edu.sg</a>)
 * 
 * @author Ye Nan
 * @author Arkadiusz Jachnik
 */
public class QuadraticNaiveFMaximizer implements Serializable {

	private static final long serialVersionUID = 4095430934497706669L;
	
	public static enum AlgorithmComplexity { QUADRATIC, CUBIC };
	
	private AlgorithmComplexity compexity = AlgorithmComplexity.QUADRATIC;

	/**
	 * Double Array comparator.
	 */
	private static class PairComp implements Comparator<double[]> {

		public PairComp() {
		}

		public int compare(double[] ar1, double[] ar2) {
			if (ar1[0] > ar2[0])
				return -1;
			if (ar1[0] < ar2[0])
				return 1;
			return 0;
		}

		@SuppressWarnings("unused")
		public boolean equals(double[] ar1, double[] ar2) {
			return ar1[0] == ar2[0];
		}
	}
	
	/**
	 * Class constructor.
	 */
	public QuadraticNaiveFMaximizer(AlgorithmComplexity ac) {
		this.compexity = ac;
	}
	
	public QuadraticNaiveFMaximizer() {
		this.compexity = AlgorithmComplexity.QUADRATIC;
	}

	/**
	 * Sort p in descending order, and return the corresponding ordering of
	 * elements in p.
	 * 
	 * @param p vector of probabilities
	 */
	private int[] sort(double[] p) {
		int N = p.length;
		
		double[][] indexed = new double[N][2];
		
		for (int i = 0; i < N; i++) {
			indexed[i][0] = p[i];
			indexed[i][1] = i;
		}
		
		Arrays.sort(indexed, new PairComp());
		
		int[] perm = new int[N];
		
		for (int i = 0; i < N; i++) {
			p[i] = indexed[i][0];
			perm[i] = (int) indexed[i][1];
		}
		
		return perm;
	}

	/**
	 * Returns maximum value in the given array of doubles
	 * 
	 * @param ar double array
	 * @return maximum value in the array
	 */
	private double maxValue(double[] ar) {
		double max = Double.NEGATIVE_INFINITY;
		
		for (double d : ar)
			if (d > max)
				max = d;
		
		return max;
	}

	/**
	 * Returns index of the maximum value in the given array
	 * 
	 * @param ar double array
	 * @return index of maximum value in the array
	 */
	private int maxIndex(double[] ar) {
		double max = Double.NEGATIVE_INFINITY;
		int index = -1;
		
		for (int i = 0; i < ar.length; i++) {
			if (ar[i] > max) {
				max = ar[i];
				index = i;
			}
		}
		
		return index;
	}

	/**
	 * Changes the indexes of values of array ar on the basis of indexes
	 * given by int array perm.
	 * 
	 * @param ar double array ar
	 * @param perm array of new indexes
	 */
	private void perm(double[] ar, int[] perm) {
		int N = ar.length;
		double[] copy = new double[N];
		
		for (int i = 0; i < N; i++)
			copy[perm[i]] = ar[i];
		
		for (int i = 0; i < N; i++)
			ar[i] = copy[i];
	}

	/**
	 * Return the maximum expected F<sub>q/r</sub> when the probabilities that
	 * the instances are positive are given by p
	 * 
	 * @param p vector of probabilities 
	 * @param q parameter q
	 * @param r parameter r
	 * @return maximum expected F<sub>q/r</sub>
	 */
	public double maxExpectedFScore(double[] p, int q, int r) {
		int[] perm = sort(p);
		double f = maxValue(expectedFScores(p, q, r));
		
		perm(p, perm);
		
		return f;
	}

	/**
	 * Return the predictions maximizing the expected F<sub>q/r</sub> when the
	 * probabilities that the instances are positive are given by p.
	 * 
	 * @param p vector of probabilities 
	 * @param q parameter q
	 * @param r parameter r
	 * @return maximum expected F<sub>q/r</sub>
	 */
	private int[] maxExpectedFScorePreds(double[] p, int q, int r) {
		int N = p.length;
		int[] perm = sort(p);
		int R = maxIndex(expectedFScores(p, q, r));
		int[] preds = new int[N];
		
		for (int i = 0; i < R; i++)
			preds[perm[i]] = 1;
		
		perm(p, perm);
		
		return preds;
	}

	/**
	 * Return the F<sub>q/r</sub> scores when the first k instances are
	 * predicted as positive, for all k.
	 * 
	 * @param p vector of probabilities 
	 * @param q parameter q
	 * @param r parameter r
	 * @return maximum expected F<sub>q/r</sub>
	 */
	private double[] expectedFScores(double[] p, int q, int r) {
		int nInsts = p.length;
		double[] fs = new double[nInsts + 1];
		double beta = 1.0 * q / r;

		@SuppressWarnings("unused")
		int A = 0, B = q + r;
		double[] sums = new double[(q + r) * nInsts + 2];
		
		for (int i = 1; i <= B * nInsts; i++)
			sums[i] = 1.0 * q / i;

		double[][] poly = new double[nInsts + 1][]; // [p_1x+(1-p_1)][p_2x+(1-p_2)]...[p_nx+(1-p_n)]
		poly[0] = new double[] { 0, 1, 0 };
		
		for (int i = 0; i < nInsts; i++) {
			poly[i + 1] = new double[i + 4];
			for (int j = i + 2; j > 0; j--) {
				poly[i + 1][j] = (1 - p[i]) * poly[i][j] + p[i]
						* poly[i][j - 1];
			}
		}

		for (int n = nInsts; n > 0; n--) {
			for (int TP = 0; TP <= n; TP++) {
				fs[n] += ((1 + beta) / beta) * TP * poly[n][TP + 1]
						* sums[n * r + TP * q];
			}
			
			for (int i = 1; i <= B * (n - 1); i++) {
				sums[i] = (1 - p[n - 1]) * sums[i] + p[n - 1] * sums[i + q];
			}
		}

		fs[0] = expectedFScore(p, 0, beta);

		return fs;
	}

	/**
	 * Return the maximum expected F<sub>beta</sub> when the probabilities that
	 * the instances are positive are given by p
	 * 
	 * @param p vector of probabilities 
	 * @param beta parameter beta
	 * @return maximum expected F<sub>beta</sub>
	 */
	public double maxExpectedFScore(double[] p, double beta) {
		int[] perm = sort(p);
		double f = maxValue(expectedFScores(p, beta));
		
		perm(p, perm);
		
		return f;
	}

	/**
	 * Return the predictions maximizing the expected F<sub>beta</sub> when the
	 * probabilities that the instances are positive are given by p.
	 * 
	 * @param p vector of probabilities 
	 * @param beta parameter beta
	 * @return maximum expected F<sub>beta</sub>
	 */
	private int[] maxExpectedFScorePreds(double[] p, double beta) {
		int N = p.length;
		int[] perm = sort(p);
		int R = maxIndex(expectedFScores(p, beta));
		int[] preds = new int[N];
		
		for (int i = 0; i < R; i++)
			preds[perm[i]] = 1;
		
		perm(p, perm);
		
		return preds;
	}

	/**
	 * Return the F<sub>beta</sub> scores when the first n instances are
	 * predicted as positive, for all n.
	 * 
	 * @param p vector of probabilities 
	 * @param beta parameter beta
	 * @return maximum expected F<sub>beta</sub>
	 */
	public double[] expectedFScores(double[] p, double beta) {
		double[] fs = new double[p.length + 1];
		
		for (int n = 0; n <= p.length; n++)
			fs[n] = expectedFScore(p, n, beta);
		
		return fs;
	}

	/**
	 * Return the F<sub>beta</sub> score when the first n instances are
	 * predicted as positive.
	 * 
	 * @param p vector of probabilities 
	 * @param n n positive predicted instances 
	 * @return maximum expected F<sub>beta</sub>
	 */
	public double expectedFScore(double[] p, int n, double beta) {
		int nInsts = p.length;

		if (n == 0) {
			double degenerateF1 = 1;
			for (int i = 0; i < nInsts; i++) {
				degenerateF1 *= (1 - p[i]);
			}
			return degenerateF1;
		}

		double[] poly1 = new double[nInsts + 2]; // [p_1x+(1-p_1)][p_2x+(1-p_2)]...[p_nx+(1-p_n)]
		poly1[1] = 1;
		
		for (int i = 0; i < n; i++) {
			for (int j = i + 2; j > 0; j--)
				poly1[j] = (1 - p[i]) * poly1[j] + p[i] * poly1[j - 1];
		}

		double[] poly2 = new double[nInsts + 2]; // [p_{n+1}x+(1-p_{n+1})]...[p_Nx+(1-p_N)]
		poly2[1] = 1;
		
		for (int i = 0; i < nInsts - n; i++) {
			for (int j = i + 2; j > 0; j--)
				poly2[j] = (1 - p[n + i]) * poly2[j] + p[n + i] * poly2[j - 1];
		}

		double E = 0;
		
		for (int TP = 0; TP <= n; TP++) {
			for (int FN = 0; FN <= nInsts - n; FN++) {
				E += Math.abs(poly1[1 + TP] * poly2[1 + FN]) * (1 + beta) * TP
						/ (n + beta * (TP + FN));
			}
		}
		
		return E;
	}

	/**
	 * Returns prediction with regard to F-measure.
	 * 
	 * @param confidences confidences for labels given by any classifier
	 * @return an array with prediction
	 */
	public boolean[] predictionForInstance(double[] confidences) {
		return predictionForInstance(confidences, 1, 1);
	}
	
	/**
	 * Returns prediction with regard to F-measure.
	 * 
	 * @param confidences confidences for labels given by any classifier
	 * @param q parameter for F<sub>q/r</sub> score
	 * @param r parameter for F<sub>q/r</sub> score
	 * @return an array with prediction
	 */
	public boolean[] predictionForInstance(double[] confidences, int q, int r) {
		int[] ranking = new int[confidences.length];
		boolean[] predictions = new boolean[confidences.length];
		
		if(this.compexity == AlgorithmComplexity.QUADRATIC) {
			ranking = maxExpectedFScorePreds(confidences, q, r);
		} else {
			double beta = 1.0 * q / r;
			ranking = maxExpectedFScorePreds(confidences, beta);
		}
		
		for(int i = 0; i < ranking.length; i++) {
			if(ranking[i] == 1)
				predictions[i] = true;
		}
		
		return predictions;
	}

}
