package put.mlc.classifiers.pcc.inference.montecarlo;

import put.mlc.classifiers.pcc.inference.IInference;
import put.mlc.classifiers.pcc.inference.common.LabelCombination;
import put.mlc.classifiers.rankloss.RankLossMaximizer;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import weka.core.Instance;

/**
 * Implementation of Ranking Loss Maximizer. This is one of inference
 * methods of PCC algorithm.
 * <br>
 * For more information, see:<br>
 * <br>
 * Krzysztof Dembczynski, Wojciech Kotlowski, Eyke Hullermeier,
 * "Consistent Multilabel Ranking through Univariate Losses", ICML 2012
 * 
 * @author Arkadiusz Jachnik
 */
public class RankLossMaximizerInference extends MonteCarloInference implements IInference {

	private static final long serialVersionUID = 1223434438266950744L;

	/**
	 * Class constructor specifying the number of simulations in Monte Carlo
	 * sampling method.
	 * 
	 * @param numOfSimulations number of simulations
	 */
	public RankLossMaximizerInference(int numSimulations, int seed) {
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
		LabelCombination[] sample = new LabelCombination[this.numSimulations];

		int index = monteCarloSampling(instance, sample);

		return computeRankLossMaximizer(sample, index); 
	}
	
	/**
	 * Computes marginal probabilities for optimization of the Ranking Loss.
	 * 
	 * @param sample array containing label combinations
	 * @param length number of label combinations
	 */
	private MultiLabelOutput computeRankLossMaximizer(LabelCombination[] sample, int length) {
		RankLossMaximizer rlm = new RankLossMaximizer(numLabels);
		
		for (int i = 0; i < length; i++) {
			double[] labels = sample[i].getCombination();
			for (int j = 0; j < sample[i].getFreq(); j++) {
				rlm.add(labels);
			}
		}
		
		double[] confidences = rlm.computeRankLoss();
		boolean[] bipartition = rlm.getBinaryPrediction();
		
		MultiLabelOutput result = new MultiLabelOutput(bipartition,confidences);
		
		return result;
	}
	
	/**
	 * Returns a string containing the name of specified inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public String getName() {
		return "Inference by Ranking Loss Optimization (numSimulations=" + numSimulations + ")";
	}

}
