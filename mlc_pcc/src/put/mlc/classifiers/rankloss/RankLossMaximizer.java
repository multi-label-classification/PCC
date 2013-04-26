package put.mlc.classifiers.rankloss;

/**
 * Implementation of the inference algorithm for the Ranking Loss Maximizer
 * in Probabilistic Classifier Chains.
 * <br>
 * For more information, see:<br>
 * <br>
 * Krzysztof Dembczynski, Wojciech Kotlowski, Eyke Hullermeier,
 * "Consistent Multilabel Ranking through Univariate Losses", ICML 2012
 * 
 * @author Arkadiusz Jachnik
 */
public class RankLossMaximizer {

	/**
	 * number of labels
	 */
	private int numOfLabels = 0;
	
	/**
	 * number of instances (or samples)
	 */
	private int numOfInstances = 0;
	
	/**
	 * weighted number of positives (1s) for each label
	 */
	private double[] counts = null;
	
	/**
	 * number of positives (1s) for each label (e.g. for Hamming Loss)
	 */
	private int[] countsHL = null;
	
	/**
	 * weighted marginal probabilities 
	 */
	private double[] marginals = null;
	
	/**
	 * vector of prediction
	 */
	private boolean[] binaryPrediction = null;
	
	/**
	 * threshold for Hamming Loss
	 */
	private double HLThreshold = 0.5;
	
	/**
	 * Default constructor.
	 * 
	 * @param numOfLabels number of labels
	 */
	public RankLossMaximizer(int numOfLabels) {
		this.numOfLabels = numOfLabels;
		this.counts = new double[this.numOfLabels];
		this.countsHL = new int[this.numOfLabels];
		this.marginals = new double[this.numOfLabels];
		
		for(int i = 0; i < this.counts.length; i++) {
			this.counts[i] = 0;
		}
	}
	
	/**
	 * Adds instance or sample.
	 * 
	 * @param prediction prediction vector of byte values
	 */
	public void add(byte[] prediction) {
		int[] p = new int[prediction.length];
		for(int i = 0; i < prediction.length; i++) {
			p[i] = (int)prediction[i];
		}
		this.add(p);
	}
	
	/**
	 * Adds instance or sample.
	 * 
	 * @param prediction prediction vector of double value
	 */
	public void add(double[] prediction) {
		int[] p = new int[prediction.length];
		for(int i = 0; i < prediction.length; i++) {
			p[i] = (int)prediction[i];
		}
		this.add(p);
	}
	
	/**
	 * Adds instance or sample.
	 * 
	 * @param prediction prediction vector of integer values
	 */
	public void add(int[] prediction) {
		int positives = 0;
		int negatives = 0;
		
		for (int l = 0; l < this.numOfLabels; l++) {
			if(prediction[l] > 0) {
				this.countsHL[l]++;
				positives++;
			} else {
				negatives++;
			}
		}
		
		double w = 1.0 / (double)(positives * negatives);
		
		if(Double.isInfinite(w))
			return;
		
		for (int l = 0; l < this.numOfLabels; l++) {
			if(prediction[l] > 0) {
				this.counts[l] += w;
			}
		}
		
		this.numOfInstances++;
	}
	
	/**
	 * Computes weighted marginal probabilities for each labels, which will
	 * be used for obtain optimal Ranking Loss.
	 * 
	 * @return vector of weighted marginal probabilities (deltas)
	 */
	public double[] computeRankLoss() {
		this.binaryPrediction = new boolean[this.numOfLabels];
		
		for (int l = 0; l < this.numOfLabels; l++) {
			double p = (double) this.countsHL[l] / (double) this.numOfInstances;
			if(p >= HLThreshold)
				this.binaryPrediction[l] = true;
			
			this.marginals[l] = (double) this.counts[l] / (double) this.numOfInstances;
		}
		
		return this.marginals.clone();
	}
	
	/**
	 * @return vector of binary prediction for Hamming Loss
	 */
	public boolean[] getBinaryPrediction() {
		return this.binaryPrediction.clone();
	}
}
