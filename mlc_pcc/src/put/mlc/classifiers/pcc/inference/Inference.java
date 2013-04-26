package put.mlc.classifiers.pcc.inference;

import java.io.Serializable;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;

/**
 * Root class for all inference methods in PCC.
 * 
 * @author Adrian Jaroszewicz
 * @author Arkadiusz Jachnik
 */
public abstract class Inference implements IInference, Serializable {
	
	private static final long serialVersionUID = 2169913300995337925L;
	
	protected Classifier[] ensemble;
	protected int[] chain;
	protected int numLabels;
	protected int[] labelIndices;
	protected boolean isMultiThreaded = false;
	
	protected Attribute[] classAttributes;
	
	public Inference() {}
	
	public Inference(int arg0) {}
	
	/**
	 * Sets the chain of binary classifiers used by PCC.
	 * 
	 * @param ensemble an array with chain of classifiers
	 */
	public void setEnsemble(Classifier[] ensemble) {
		this.ensemble = ensemble;
	}

	/**
	 * Sets an array containing indexes of labels.
	 * 
	 * @param chain an array with indexes of all labels
	 */
	public void setChain(int[] chain) {
		this.chain = chain;
	}

	/**
	 * @return an array containing indexes of labels
	 */
	public int[] getChain() {
		return chain;
	}

	/**
	 * Sets the number of labels in specified data set.
	 * 
	 * @param numLabels number of labels
	 */
	public void setNumLabels(int numLabels) {
		this.numLabels = numLabels;
	}

	/**
	 * Sets an array containing indexes of labels in the list of attributes.
	 * 
	 * @param labelIndices an array containing indexes of labels
	 */
	public void setLabelIndices(int[] labelIndices) {
		this.labelIndices = labelIndices;
	}

	/**
	 * @return an array containing indexes of labels
	 */
	public int[] getLabelIndices() {
		return labelIndices;
	}

	/**
	 * Sets an array with class attributes.
	 * 
	 * @param classAttributes an array with class attributes
	 */
	public void setClassAttributes(Attribute[] classAttributes) {
		this.classAttributes = classAttributes;
	}
	
	
	/**
	 * Converts double to boolean. If the given number is higher or equal
	 * to 0.5, it returns true. False otherwise.
	 * 
	 * @param doubles an array with doubles to convert
	 * @return an array with booleans
	 */
	protected boolean[] booleansFromDoubles(double[] doubles) {
		return booleansFromDoubles(doubles, 0.5);
	}
	
	/**
	 * Converts double to boolean. If the given number is higher or equal
	 * to 't', it returns true. False otherwise.
	 * 
	 * @param doubles an array with doubles to convert
	 * @param t threshold
	 * @return an array with booleans
	 */
	public boolean[] booleansFromDoubles(double[] doubles, double t) {

		boolean[] labels = new boolean[doubles.length];
		for (int i = 0; i < doubles.length; i++) {
			labels[i] = (doubles[i] >= t);
		}

		return labels;
	}
	
	/**
	 * @return the state of multi-threading flag
	 */
	public boolean isMultiThreaded() {
		return isMultiThreaded;
	}

	/**
	 * @param isMultiThreaded the multi-threading flag to set
	 */
	public void setMultiThreaded(boolean isMultiThreaded) {
		this.isMultiThreaded = isMultiThreaded;
	}
	
	/**
	 * Runs an inference procedure for a given instance.
	 * 
	 * @param instance instance to classify
	 * @return output of a {@link MultiLabelLearner}
	 * @throws Exception
	 */
	@Override
	public abstract MultiLabelOutput inferenceProcedure(Instance instance) throws Exception;

	/**
	 * Returns a string containing the name of specified inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public abstract String getName();
}
