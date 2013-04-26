package put.mlc.classifiers.pcc.inference.common;

/**
 * Implementation of a representation of some label combination.
 * 
 * @author Arkadiusz Jachnik
 */
public class LabelCombination {

	/**
	 * index of currently observed label
	 */
	protected int currentLabel = 0;
	
	/**
	 * probability
	 */
	protected double p = 1.0;
	
	/**
	 * value of last saved probability
	 */
	protected double lastP = 1.0;
	
	/**
	 * labels combination
	 */
	protected double[] labels = null;
	
	/**
	 * number of visits the node (when the label combination is a node of tree)
	 */
	protected int freq = 0;

	public LabelCombination() {

	}

	public LabelCombination(LabelCombination copy) {
		this.copy(copy);
	}

	public void copy(LabelCombination copy) {
		this.currentLabel = copy.currentLabel;
		this.p = copy.p;
		this.lastP = copy.lastP;
		this.freq = copy.freq;

		if (copy.labels != null) {
			this.labels = new double[copy.labels.length];
			for (int i = 0; i < this.currentLabel; i++) {
				this.labels[i] = copy.labels[i];
			}
		}
	}

	public double getP() {
		return this.p;
	}

	public void setP(double p) {
		this.p = p;
	}

	public void setNextLabel(double p) {
		this.currentLabel++;
		this.p *= p;
		this.lastP = p;
	}

	public double[] getCombination() {
		return this.labels;
	}

	public void setCombination(double[] labels) {
		this.labels = labels;
	}

	public int getCurrentLabel() {
		return this.currentLabel;
	}

	public double getLastP() {
		return lastP;
	}

	public void increaseFreq() {
		this.freq++;
	}

	public void resetFreq() {
		this.freq = 0;
	}

	public int getFreq() {
		return this.freq;
	}

	public String toString() {
		String str = "";
		for (int i = 0; i < this.currentLabel; i++) {
			str += (int) this.labels[i];
		}

		str += "\t" + this.freq;

		return str;
	}

}
