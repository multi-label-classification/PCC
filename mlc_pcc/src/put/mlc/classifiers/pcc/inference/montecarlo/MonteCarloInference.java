package put.mlc.classifiers.pcc.inference.montecarlo;

import java.util.Random;

import put.mlc.classifiers.pcc.inference.Inference;
import put.mlc.classifiers.pcc.inference.common.LabelCombination;
import put.mlc.classifiers.pcc.inference.common.LabelCombinationTree;
import mulan.data.DataUtils;
import weka.core.Attribute;
import weka.core.Instance;

/**
 * Implementation of the Monte Carlo Inference algorithm 
 * in Probabilistic Classifier Chains.<br>
 * The algorithm samples arrays of label combinations with the Monte Carlo.
 * The Monte Carlo method is used for estimation of probability distribution.
 * 
 * @author Krzysztof Dembczynski
 */
public abstract class MonteCarloInference extends Inference {

	private static final long serialVersionUID = 5651868930083557543L;

	/**
	 * number of simulations for sampling procedure
	 */
	int numSimulations = 100;
	
	/**
	 * seed value
	 */
	int seed = 1; 
	
	/**
	 * Class constructor.
	 * 
	 * @param numSimulations number of simulations
	 * @param seed value of the seed
	 */
	public MonteCarloInference(int numSimulations, int seed) {
		this.numSimulations = numSimulations;
		this.seed = seed;
	}

	/**
	 * Sets the number of simulations in Monte Carlo sampling.<br>
	 * Monte Carlo method is used for estimation of probability
	 * distribution.
	 * 
	 * @param numOfSimulations number of simulations
	 */
	public void setNumOfSimulations(int numOfSimulations) {
		this.numSimulations = numOfSimulations;
	}
	
	/**
	 * Returns the number of simulations.
	 * @return number of simulations
	 */
	public int getNumOfSimulations() {
		return numSimulations;
	}

	/**
	 * Sets the seed used in Monte Carlo method.
	 * 
	 * @param seed seed for randomization
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	/**
	 * Returns an array containing class attributes.
	 * 
	 * @return class attributes
	 */
	public Attribute[] getClassAttributes() {
		return classAttributes;
	}

	/**
	 * Sampling method for label combinations generation with the Monte Carlo
	 * method.
	 * 
	 * @param instance instance to classify
	 * @param sample array of label combinations that is going to be filled
	 * @return number of label combinations
	 * @throws Exception
	 */
	protected int monteCarloSampling(Instance instance,
			LabelCombination[] sample) throws Exception {

		Random random = new Random(this.seed);
		
		Instance tempInstance = DataUtils.createInstance(instance, instance.weight(),
				instance.toDoubleArray());
		
		LabelCombinationTree root = new LabelCombinationTree(new LabelCombination());
		
		int length = 0;
		
		for (int s = 0; s < numSimulations; s++) {
			LabelCombinationTree current = root;

			double[] values = new double[numLabels];
			
			for (int i = 0; i < this.numLabels; i++) {
				double[] p;

				if (current.hasKids()) {
					p = new double[2];
					p[0] = current.left().root().getLastP();
					p[1] = current.right().root().getLastP();
				}
				else {
					p = this.ensemble[i].distributionForInstance(tempInstance);
					
					LabelCombination left = new LabelCombination(current.root());
					//left.resetState();
					left.resetFreq();
					left.setNextLabel(p[0]);
					LabelCombinationTree leftTree = new LabelCombinationTree(left);

					LabelCombination right = new LabelCombination(current.root());
					//right.resetState();
					right.resetFreq();
					right.setNextLabel(p[1]);
					LabelCombinationTree rightTree = new LabelCombinationTree(right);

					current.setKids(leftTree, rightTree);
				}

				int y_i = (p[1] > random.nextDouble()) ? 1 : 0;
				values[i] = y_i;
				tempInstance.setValue(labelIndices[chain[i]], y_i);
				
				if (y_i == 1)
					current = current.right();
				else
					current = current.left();
				
				current.root().increaseFreq();
			}

			if (current.root().getFreq() == 1) {
				current.root().setCombination(values);
				sample[length] = current.root();
				length++;
			}
		}
		
		return length;
	}
	
	/**
	 * Computes marginal probabilities from a specified path in decision tree.
	 * 
	 * @param confidences array with marginal probabilities that is going to
	 * be filled
	 * @param sample path in decision tree
	 * @param length
	 */
	protected void computeMarginals(double[] confidences,
			LabelCombination[] sample, int length) {
		int sum = 0;

		for (int i = 0; i < length; i++) {
			double[] labels = sample[i].getCombination();

			for (int j = 0; j < labels.length; j++) {
				confidences[j] += ((int) labels[j] * (int) sample[i].getFreq());
			}
			sum += sample[i].getFreq();
		}

		for (int j = 0; j < confidences.length; j++)
			confidences[j] /= (double) sum;
	}
	
	/**
	 * Fills the array of predictions on the basis of modes from the samples' array.
	 *  
	 * @param predictions array of predictions to be filled
	 * @param sample array of samples of label combinations
	 * @param length number of label combinations
	 */
	protected void computeMode(double[] predictions, LabelCombination[] sample,
			int length) {

		int mode = 0;
		int max = 0;

		for (int i = 0; i < length; i++) {
			if (sample[i].getFreq() > max) {
				max = sample[i].getFreq();
				mode = i;
			}
		}

		for (int i = 0; i < predictions.length; i++) {
			predictions[i] = sample[mode].getCombination()[i];
		}

	}
	
	/**
	 * Returns a string containing the name of this inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public String getName() {
		return "Monte Carlo inference";
	}

}
