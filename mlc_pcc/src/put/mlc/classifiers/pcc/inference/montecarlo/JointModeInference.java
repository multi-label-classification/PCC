package put.mlc.classifiers.pcc.inference.montecarlo;

import put.mlc.classifiers.pcc.inference.common.LabelCombination;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import weka.core.Instance;

/**
 * Implementation of the Joint Mode Inference algorithm 
 * in Probabilistic Classifier Chains. It is based on the Monte
 * Carlo sampling.
 * 
 * @author Krzysztof Dembczynski
 */
public class JointModeInference extends MonteCarloInference {

	private static final long serialVersionUID = 5651868930083557543L;

	/**
	 * Class constructor.
	 * 
	 * @param numOfSimulations number of simulations
	 * @param seed seed value
	 */
	public JointModeInference(int numOfSimulations, int seed) {
		super(numOfSimulations, seed);
	}

	/**
	 * Runs an inference procedure for a given instance.
	 * 
	 * @param instance instance to classify
	 * @return output of a {@link MultiLabelLearner}
	 * @throws Exception
	 */
	@Override
	public MultiLabelOutput inferenceProcedure(Instance instance)
			throws Exception {
		double[] confidences = new double[this.numLabels];
		double[] predictions = new double[this.numLabels];

		LabelCombination[] sample = new LabelCombination[this.numSimulations];

		int index = monteCarloSampling(instance, sample);

		computeMarginals(confidences, sample, index);
		computeMode(predictions, sample, index);

		MultiLabelOutput result = new MultiLabelOutput(
				booleansFromDoubles(predictions), confidences);

		return result;
	}
	
	/**
	 * Returns a string containing the name of this inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public String getName() {
		return "Joint Mode Monte Carlo inference " + numSimulations;
	}

}
