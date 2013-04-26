package put.mlc.classifiers.pcc.inference.montecarlo;

import put.mlc.classifiers.pcc.inference.common.LabelCombination;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import weka.core.Instance;

/**
 * Implementation of the Marginal Modes Inference algorithm 
 * in Probabilistic Classifier Chains. It is based on the Monte
 * Carlo sampling.
 * 
 * @author Krzysztof Dembczynski
 */
public class MarginalModesInference extends MonteCarloInference {

	private static final long serialVersionUID = 5651868930083557543L;

	public MarginalModesInference(int numOfSimulations, int seed) {
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
		
		LabelCombination[] sample = new LabelCombination[this.numSimulations];

		int index = monteCarloSampling(instance, sample);

		computeMarginals(confidences, sample, index);
		
		MultiLabelOutput result = new MultiLabelOutput(
				booleansFromDoubles(confidences), confidences);

		return result;
	}
	
	/**
	 * Returns a string containing the name of this inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public String getName() {
		return "Marginal Modes Monte Carlo inference\t" + numSimulations;
	}

}
