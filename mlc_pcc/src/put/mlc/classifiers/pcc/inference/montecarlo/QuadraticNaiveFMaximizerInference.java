package put.mlc.classifiers.pcc.inference.montecarlo;

import put.mlc.classifiers.f.QuadraticNaiveFMaximizer;
import put.mlc.classifiers.f.QuadraticNaiveFMaximizer.AlgorithmComplexity;
import put.mlc.classifiers.pcc.inference.common.LabelCombination;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import weka.core.Instance;

/**
 * Implementation of the Quadratic Naive Inference method 
 * in Probabilistic Classifier Chains. It is based on the Monte
 * Carlo sampling.<br>
 * 
 * This class uses {@link QuadraticNaiveFMaximizer} - quadratic time algorithm 
 * for computing the predictions maximizing the expected F-measure under the 
 * label independence assumption. The algorithm is described in <i>Nan Ye, 
 * Adam K.M. Chai, Wee Sun Lee, Hai Leong Chieu. Optimizing F-measures: A Tale 
 * of Two Approaches. ICML 2012.</i>
 * 
 * @author Arkadiusz Jachnik
 * 
 * @see {@link QuadraticNaiveFMaximizer}
 */
public class QuadraticNaiveFMaximizerInference extends MonteCarloInference {

	private static final long serialVersionUID = 3223048186819905661L;

	/**
	 * Class constructor.
	 * 
	 * @param numSimulations number of simulations
	 * @param seed seed value
	 */
	public QuadraticNaiveFMaximizerInference(int numSimulations, int seed) {
		super(numSimulations, seed);
	}

	/**
	 * Runs an inference procedure for a given instance.
	 * 
	 * @param instance instance to classify
	 * @return output of a {@link MultiLabelLearner}
	 * @throws Exception
	 */
	@Override
	public MultiLabelOutput inferenceProcedure(Instance instance) throws Exception {
		double[] confidences = new double[this.numLabels];

		LabelCombination[] sample = new LabelCombination[this.numSimulations];

		int index = monteCarloSampling(instance, sample);

		computeMarginals(confidences, sample, index);
		
		QuadraticNaiveFMaximizer qta = 
				new QuadraticNaiveFMaximizer(AlgorithmComplexity.QUADRATIC);
		boolean[] bipartition = qta.predictionForInstance(confidences);
		MultiLabelOutput result = new MultiLabelOutput(bipartition);

		return result;
	}
	
	/**
	 * Returns a string containing the name of this inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public String getName() {
		return "F-measure inference by Ye's Quadratic Time Algorithm " + numSimulations;
	}

}
