package put.mlc.classifiers.pcc.inference;

import java.util.LinkedList;
import put.mlc.classifiers.pcc.inference.common.LabelCombinationExtended;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import weka.core.Instance;

/**
 * Implementation of the Exhaustive Inference algorithm 
 * in Probabilistic Classifier Chains.
 * 
 * @author Krzysztof Dembczynski
 */
public class ExhaustiveInference extends Inference {

	private static final long serialVersionUID = 3649436749916126605L;
	
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
		LabelCombinationExtended best = null;

		Instance tempInstance = DataUtils.createInstance(instance,
				instance.weight(), instance.toDoubleArray());
		
		LinkedList<LabelCombinationExtended> lifo = new LinkedList<LabelCombinationExtended>();
		lifo.add(new LabelCombinationExtended(this.numLabels, tempInstance, this));

		while (!lifo.isEmpty()) {

			LabelCombinationExtended current = lifo.getFirst();
			int label = current.nextState();
			if (label > 1 || current.getP() == 0) {
				lifo.removeFirst();
				continue;
			}

			Instance currentInstance = current.getInstance();
			
			int i = current.getCurrentLabel();
			
			double p = this.ensemble[i].distributionForInstance(currentInstance)[label];

			LabelCombinationExtended next = new LabelCombinationExtended(current);
			next.resetState();
			next.setNextLabel(label, p);

			if (next.getCurrentLabel() == this.numLabels) {
				if (best == null || next.getP() > best.getP()) {
					best = new LabelCombinationExtended(next);
				}
				for (int s = 0; s < this.numLabels; s++) {
					if (next.getCombination()[s] == 1)
						confidences[s] += next.getP();
				}
			} else {
				lifo.addFirst(next);
			}
		}

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
		return "Exhaustive Inference";
	}

}
