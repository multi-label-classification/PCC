package put.mlc.classifiers.pcc.inference.common;

import put.mlc.classifiers.pcc.inference.Inference;
import weka.core.Instance;
import weka.core.SparseInstance;

/**
 * Implementation of a representation of some label combination.
 * This is extended version of {@link LabelCombination} class.
 * It additionally contains information about inference.
 * 
 * @author Arkadiusz Jachnik
 */
public class LabelCombinationExtended extends LabelCombination {

	/**
	 * number of labels
	 */
	protected int numLabels = 0;
	
	/**
	 * state
	 */
	protected int state = -1;
	
	/**
	 * Weka Instance
	 */
	protected Instance instance = null;
	
	/**
	 * reference to the inference procedure
	 */
	protected Inference inference = null;

	public LabelCombinationExtended() {
		super();
	}

	public LabelCombinationExtended(LabelCombinationExtended copy) {
		this.copy(copy);
	}
	
	public LabelCombinationExtended(int numLabels, Instance instance, Inference inference) {
		this.numLabels = numLabels;
		this.labels = new double[this.numLabels];
		this.inference = inference;
		this.instance = (Instance) instance.copy();
	}

	public void copy(LabelCombinationExtended copy) {
		super.copy(copy);
		this.numLabels = copy.numLabels;
		this.state = copy.state;
		this.instance = (Instance) copy.getInstance().copy();
	}

	public void setNextLabel(int prediction, double p) {
		this.labels[this.currentLabel] = prediction;
		this.currentLabel++;
		this.p *= p;
		this.lastP = p;

		if (currentLabel < this.numLabels) {
			instance.setValue(
					this.inference.getLabelIndices()[this.inference.getChain()[currentLabel - 1]],
					prediction);
		}
	}
	
	public int nextState() {
		return ++this.state;
	}

	public int getNumLabels() {
		return numLabels;
	}

	public void setNumLabels(int numLabels) {
		this.numLabels = numLabels;
	}

	public int getState() {
		return state;
	}

	public void setState(int state) {
		this.state = state;
	}
	
	public void resetState() {
		this.state = -1;
		this.freq = 0;
	}

	public Instance getInstance() {
		return instance;
	}

	public void setInstance(Instance instance) {
		this.instance = instance;
	}

	public Inference getInference() {
		return inference;
	}

	public void setInference(Inference inference) {
		this.inference = inference;
	}

}
