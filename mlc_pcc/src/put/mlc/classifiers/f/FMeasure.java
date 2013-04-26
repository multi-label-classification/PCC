package put.mlc.classifiers.f;

import java.util.Arrays;
import put.mlc.utils.SelectionSort;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;

/**
 * Implementation of the core algorithm for the General F-Measure Maximizer.
 * <br>
 * For more information, see:<br>
 * <br>
 * Krzysztof Dembczynski, Willem Waegeman, Weiwei Cheng, Eyke Hullermeier,
 * "An exact algorithm for F-measure maximization",
 * Advances in Neural Information Processing Systems 24 (NIPS-11): 223-230.<br>
 * 
 * @author Krzysztof Dembczynski
 * @author Arkadiusz Jachnik
 */
public class FMeasure {

	/**
	 * best F-measure
	 */
	protected double fMeasure = 0;
	
	/**
	 * indexes of positive labels
	 */
	protected int[] fMaximizer = null;
	
	/**
	 * probability p(Y=0)
	 */
	protected double p_0 = 0.0;
	
	/**
	 * maximum number of relevant labels
	 */
	protected int maxNumOfRelevantLabels;
	
	/**
	 * number of sample predictions with zero positive labels
	 */
	protected int nulls = 0;
	
	/**
	 * number of labels
	 */
	protected int numLabels;
	
	/**
	 * number of instances
	 */
	protected int numOfInstances = 0;
	
	/**
	 * matrix F (or Delta)
	 */
	double[][] partialFMeasures = null;
	
	/**
	 * temporarily stores indexes of relevant labels
	 */
	private int[] temp = null; 
	
	/**
	 * the mode of selection function:
	 * if true, sort only indexes of the k-th column of matrix F;
	 * otherwise sort indexes and elements
	 */
	private boolean sortIndexes = true;
	
	/**
	 * @param p_0 the value of probability p(Y=0) to set
	 */
	public void setP_0(double p_0) {
		this.p_0 = p_0;
	}
	
	/**
	 * Compute F-measure for prediction of a given instance
	 * 
	 * @param actual the truth labels
	 * @param predicted the predicted labels
	 * @return value of computed F-measure
	 */
	public static double computeFMeasure(byte[] actual, byte[] predicted) {

		int tp = 0, y = 0, h = 0;
		for (int k = 0; k < actual.length; k++) {
			y += actual[k];
			h += predicted[k];
			if (actual[k] == 1 && predicted[k] == 1)
				tp++;
		}

		if (y + h == 0)
			return 1;
		else
			return ((double) tp * 2.0) / ((double) y + h);
	}

	/**
	 * @return the F-measure
	 */
	public double getFMeasure() {
		return this.fMeasure;
	}

	/**
	 * @return the number of instances
	 */
	public int getNumberOfInstances() {
		return this.numOfInstances;
	}

	/**
	 * Sets the initialization parameters with the same number of labels
	 * and the maximum number of relevant labels.
	 * 
	 * @param numLabels the number of labels
	 */
	public void initialize(int numLabels) {
		this.initialize(numLabels, numLabels);
	}
	
	/**
	 * Sets the initialization parameters.
	 * 
	 * @param numLabels the number of labels
	 * @param maxNumOfRelevantLabels the maximum number of relevant labels
	 */
	public void initialize(int numLabels, int maxNumOfRelevantLabels) {
		this.numLabels = numLabels;
		this.maxNumOfRelevantLabels = maxNumOfRelevantLabels;
		this.nulls = 0;
		this.p_0 = 0.0;
		this.partialFMeasures = new double[this.numLabels][this.numLabels];
		this.numOfInstances = 0;
		
		temp = new int[numLabels];  
	}

	/**
	 * Sets the initialization parameters with the initialization of the
	 * matrix F.
	 * 
	 * @param numLabels the number of labels
	 * @param maxRelevantLabels the maximum number of relevant labels
	 * @param probabilities matrix P
	 * @param p0 probability p(Y=0)
	 */
	public void initialize(int numLabels, int maxRelevantLabels, double[][] probabilities, double p0) {

		this.numLabels = numLabels;
		this.maxNumOfRelevantLabels = maxRelevantLabels;
		this.numOfInstances = 1;
		this.p_0 = p0;
		this.nulls = 0;
		this.partialFMeasures = new double[this.numLabels][this.numLabels];
		
		for (int i = 0; i < partialFMeasures.length; i++) {
			for (int j = 0; j < partialFMeasures[i].length; j++) {
				for (int k = 0; k < this.maxNumOfRelevantLabels; k++) {
					partialFMeasures[i][j] += probabilities[k][j]/(i + k + 2);
				}
			}
		}
	}

	/**
	 * Computes and fills matrix F with a given sample of prediction for
	 * a given instance.
	 * 
	 * @see <code>add(byte[] prediction)</code>
	 * @param prediction the sample of prediction in the double array
	 */
	public void add(double[] prediction) {

		byte [] t = new byte[prediction.length];
		
		for(int i = 0; i < t.length; i++) {
			t[i] = (byte) prediction[i];
		}
        
		add(t);
	}

	/**
	 * Computes and fills matrix F with a given sample of prediction for
	 * a given instance.
	 * 
	 * @see <code>add(byte[] prediction)</code>
	 * @param prediction the sample of prediction in the integer array
	 */
	public void add(int[] prediction) {
		
		byte [] t = new byte[prediction.length];
		
		for(int i = 0; i < t.length; i++) {
			t[i] = (byte) prediction[i];
		}
        
		add(t);
	}

	/**
	 * Computes and fills matrix F with a given sample of prediction for
	 * a given instance.
	 * 
	 * @param prediction the sample of prediction in the byte array
	 */
	public void add(byte[] prediction) {
		this.numOfInstances++;

		int relevantLabels = 0;
		for (int i = 0; i < prediction.length; i++) {
			if (prediction[i] > 0)
				temp[relevantLabels++] = i;
		}

		if (relevantLabels == 0) {
			this.nulls++;
		} else {
			for (int i = 0; i < relevantLabels; i++) {
				for (int j = 0; j < partialFMeasures[temp[i]].length; j++) {
					partialFMeasures[j][temp[i]] += (double) 1/ (double) (relevantLabels + j + 1);
				}
			}
		}
	}
	
	/**
	 * @return the computeFMeasureMaximizer for matrix F
	 */
	public double computeFMeasureMaximizer() {
		return computeFMeasureMaximizer(this.partialFMeasures);
	}
	
	/**
	 * Performs the maximization of the F-measure. For more information, see
	 * Algorithm 1 - General F-measure Maximizer in:
	 * Krzysztof Dembczynski, Willem Waegeman, Weiwei Cheng, Eyke Hullermeier,
	 * "An exact algorithm for F-measure maximization".
	 * 
	 * @param partialFMeasures previously calculated matrix F=PW
	 * @return the F-Measure value
	 */
	protected double computeFMeasureMaximizer(double[][] partialFMeasures) {

		if(this.nulls > 0) 
			p_0 = (double) this.nulls / (double) this.numOfInstances; 
		this.fMeasure = p_0;
		
		for (int i = 0; i < numLabels; i++) {
			double [] copy = Arrays.copyOf(partialFMeasures[i], numLabels);
			int [] index = new int[numLabels];
			
			for(int j = 0; j < index.length; j++) index[j] = j;
				
			if(sortIndexes)
				SelectionSort.selectIndexes(copy, index, i);
			else
				SelectionSort.select(copy, index, i);
			
			double sum = 0;
			for (int j = 0; j <= i; j++) {
				if(sortIndexes)
					sum += copy[index[j]];
				else
					sum += copy[j]; 
			}
			sum = 2.0 *sum/this.numOfInstances;
			
			if (sum > this.fMeasure) {
				this.fMeasure = sum;
				this.fMaximizer = Arrays.copyOf(index, i + 1); 
			}
		}

		return this.fMeasure;
	}
	
	/**
	 * @return prediction in the {@link MultiLabelLearner} format
	 */
	public MultiLabelOutput computePrediction() {
		return makePrediction(this.fMaximizer);
	}

	/**
	 * Prepares prediction in the {@link MultiLabelLearner} format.
	 * 
	 * @param t array of indexes of ones
	 * @return the {@link MultiLabelLearner} object
	 */
	private MultiLabelOutput makePrediction(int[] t) {
		double[] confidences = new double[numLabels];
		boolean[] predictions = new boolean[numLabels];

		if(t != null) {
			for (int i : t) {
				predictions[i] = true;
				confidences[i] = 1;
			}
		}

		return new MultiLabelOutput(predictions, confidences);
	}


}
