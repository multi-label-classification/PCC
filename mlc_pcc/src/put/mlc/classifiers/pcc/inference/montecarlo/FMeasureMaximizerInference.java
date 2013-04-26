package put.mlc.classifiers.pcc.inference.montecarlo;

import put.mlc.classifiers.f.FMeasure;
import put.mlc.classifiers.pcc.inference.Inference;
import put.mlc.classifiers.pcc.inference.common.LabelCombination;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import weka.core.Instance;

/**
 * Implementation of General F-Measure Maximizer. This is one of inference
 * methods of PCC algorithm. It is based on the Monte Carlo sampling.<br>
 * <br>
 * For more information, see:<br>
 * <br>
 * Krzysztof Dembczynski, Willem Waegeman, Weiwei Cheng, Eyke Hullermeier,
 * "An exact algorithm for F-measure maximization",
 * Advances in Neural Information Processing Systems 24 (NIPS-11): 223-230.<br>
 * 
 * @author Krzysztof Dembczynski
 * @author Arkadiusz Jachnik
 * @author Adrian Jaroszewicz
 */
public class FMeasureMaximizerInference extends MonteCarloInference {

	private static final long serialVersionUID = 1223434438266950744L;

	/**
	 * Class constructor specifying the number of simulations in Monte Carlo
	 * sampling method.
	 * 
	 * @param numOfSimulations number of simulations
	 */
	public FMeasureMaximizerInference(int numSimulations, int seed) {
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

		return computeFMaximizer(sample, index);
	}
	
	/**
	 * Computes prediction optimizing the F-measure.
	 * 
	 * @param sample array containing label combinations
	 * @param length number of label combinations
	 */
	private MultiLabelOutput computeFMaximizer(LabelCombination[] sample, int length) {

		FMeasure fm = new FMeasure();
		//fm.sortIndexes = true;
		fm.initialize(numLabels);

		for (int i = 0; i < length; i++) {
			double[] labels = sample[i].getCombination();
			for (int j = 0; j < sample[i].getFreq(); j++) {
				fm.add(labels);
			}
		}

		fm.computeFMeasureMaximizer();
		
		return fm.computePrediction();
	}
	
	/**
	 * Returns a string containing the name of this inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public String getName() {
		return "Inference by General F-Measure Maximizer (numSimulations=" + numSimulations + ")";
	}

}
