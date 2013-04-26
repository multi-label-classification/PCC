package put.mlc.classifiers.pcc.inference.depthfirst;

import java.util.Comparator;
import java.util.PriorityQueue;

import put.mlc.classifiers.pcc.inference.Inference;
import put.mlc.classifiers.pcc.inference.common.LabelCombinationExtended;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import weka.core.Instance;

/**
 * Implementation of the inference for Probabilistic Classifier Chains.
 * It is based on the Depth First Exploration algorithm.
 * 
 * @author Krzysztof Dembczynski
 */
public class DepthFirstExplorationInference extends Inference {
	
	private static final long serialVersionUID = 4059288932944058248L;
	
	private double max = 0.0;

	public double getMax() {
		return max;
	}

	public void setMax(double max) {
		this.max = max;
	}

	public DepthFirstExplorationInference() {
		super();
	}
	
	public DepthFirstExplorationInference(double max) {
		setMax(max);
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

		Instance tempInstance = DataUtils.createInstance(instance,
				instance.weight(), instance.toDoubleArray());
		
		Comparator<LabelCombinationExtended> probabilityComparator = new Comparator<LabelCombinationExtended>() {

			public int compare(LabelCombinationExtended left, LabelCombinationExtended right) {
				double probabilityLeft = left.getP();
				double probabilityRight = right.getP();

				if (probabilityLeft > probabilityRight) {
					return -1;
				} else if (probabilityRight > probabilityLeft) {
					return +1;
				} else { // equal
					return 0;
				}

			}
		};

		PriorityQueue<LabelCombinationExtended> queue = new PriorityQueue<LabelCombinationExtended>(
				this.numLabels, probabilityComparator);
		queue.add(new LabelCombinationExtended(this.numLabels, tempInstance, this));

		PriorityQueue<LabelCombinationExtended> unsurvived = new PriorityQueue<LabelCombinationExtended>(
				this.numLabels, probabilityComparator);

		// double max = minMode;
		LabelCombinationExtended best = null;

		while (!queue.isEmpty()) {

			LabelCombinationExtended current = queue.poll();

			best = current;
			if (best.getCurrentLabel() == this.numLabels) {
				unsurvived.clear(); // the optimal solution has been found
				break;
			}

			Instance currentInstance = current.getInstance();
			
			int i = current.getCurrentLabel();
			
			double p = this.ensemble[i].distributionForInstance(currentInstance)[1];

			LabelCombinationExtended left = new LabelCombinationExtended(current);
			left.setNextLabel(0, 1 - p);
			boolean leftAdded = addToQueue(queue, left);

			LabelCombinationExtended right = new LabelCombinationExtended(current);
			right.setNextLabel(1, p);
			boolean rightAdded = addToQueue(queue, right);

			if (!leftAdded && !rightAdded) {
				unsurvived.add(current);
			}
		}

		this.max = 0.0;
		
		while (!unsurvived.isEmpty()) { // search for approximate solution
			LabelCombinationExtended greedy = unsurvived.poll();
			if (greedy.getP() <= this.max)
				break;
			greedy = greedyApproximation(greedy, this.max);
			if (greedy.getP() > this.max) {
				best = greedy;
				max = best.getP();
			}
		}

		MultiLabelOutput result = new MultiLabelOutput(
				booleansFromDoubles(best.getCombination())); 

		return result;
	}
	
	/**
	 * Greedy algorithm for the generation of approximate solution of label combination.
	 * 
	 * @param lc label combination
	 * @param max max threshold for distribution
	 * @return label combination
	 * @throws Exception
	 */
	private LabelCombinationExtended greedyApproximation(LabelCombinationExtended lc, double max)
			throws Exception {

		while (lc.getCurrentLabel() < this.numLabels && lc.getP() > max) {
			Instance currentInstance = lc.getInstance();
			
			int i = lc.getCurrentLabel();
			
			double p = this.ensemble[i].distributionForInstance(currentInstance)[1];
			lc.setNextLabel(p >= 0.5 ? 1 : 0, Math.max(p, 1 - p));
		}

		return lc;
	}
	
	/**
	 * Adds the given label combination to the priority queue.
	 * 
	 * @param queue priority queue of the label combinations
	 * @param lc label combination
	 * @return if sucessfull return true, otherwise false
	 */
	private boolean addToQueue(PriorityQueue<LabelCombinationExtended> queue,
			LabelCombinationExtended lc) {
		if (lc.getP() > this.max) {
			queue.add(lc);
			return true;
		} else
			return false;
	}
	
	/**
	 * Returns a string containing the name of this inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public String getName() {
		return "Dijkstra Inference";
	}
}
