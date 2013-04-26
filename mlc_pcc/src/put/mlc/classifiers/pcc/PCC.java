package put.mlc.classifiers.pcc;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import put.mlc.classifiers.pcc.inference.Inference;
import put.mlc.classifiers.pcc.inference.depthfirst.ExactInference;
import put.mlc.classifiers.pcc.inference.montecarlo.MonteCarloInference;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Implementation of the PCC (Probabilistic Classifier Chains) algorithm
 * with several inference methods.<br>
 * *
 * For more information:
 * <pre>
 * Krzysztof Dembczynski, Weiwei Cheng, Eyke Hullermeier:
 * Bayes Optimal Multilabel Classification via
 * Probabilistic Classifier Chains. ICML 2010: 279-286
 * </pre>
 * 
 * @author Krzysztof Dembczynski
 * @author Adrian Jaroszewicz
 * @author Arkadiusz Jachnik
 */
public class PCC extends TransformationBasedMultiLabelLearner {

	private static final long serialVersionUID = 5342355436L;

	/**
	 * base classifier
	 */
	private Classifier baseClassifier = null;
	
	/**
	 * the ensemble of binary relevance models
	 */
	private FilteredClassifier[] ensemble = null;
	
	/**
	 * the new chain ordering of the label indices
	 */
	private int[] chain = null;
	
	/**
	 * inference algorithm
	 */
	private Inference inference;

	/**
	 * Class constructor. When no inference method is specified, PCC runs
	 * with ExactInference.
	 * 
	 * @see ExactInference
	 */
	public PCC() {
		this.inference = new ExactInference();
	}
	
	/**
	 * Class constructor setting the inference method for PCC.
	 * 
	 * @param inference object representing an inference method
	 */
	public PCC(Inference inference) {
		setInference(inference);
	}

	/**
	 * @param inference algorithm to be set
	 */
	public void setInference(Inference inference) {
		this.inference = inference;
		inference.setChain(chain);
		inference.setEnsemble(ensemble);
		inference.setLabelIndices(labelIndices);
		inference.setNumLabels(numLabels);
	}
	
	/**
	 * Sets the base, binary classifier.
	 * @param baseClassifier binary classifier
	 */
	public void setBaseClassifier(Classifier baseClassifier) {
		this.baseClassifier = baseClassifier;
	}

	/**
     * Learner specific implementation of building the model from {@link MultiLabelInstances}
     * training data set.
     *
     * @param train the training data set
     * @throws Exception if learner model was not created successfully
     */
	@Override
	public void buildInternal(MultiLabelInstances train) throws Exception {

		if (chain == null) {
			chain = new int[numLabels];
			for (int i = 0; i < numLabels; i++) {
				chain[i] = i;
			}
		}

		Instances trainDataset;
		numLabels = train.getNumLabels();
		ensemble = new FilteredClassifier[numLabels];
		trainDataset = train.getDataSet();

		for (int i = 0; i < numLabels; i++) {
			ensemble[i] = new FilteredClassifier();
			ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

			// Indices of attributes to remove first removes numLabels
			// attributes
			// the numLabels - 1 attributes and so on.
			// The loop starts from the last attribute.
			int[] indicesToRemove = new int[numLabels - 1 - i];
			int counter2 = 0;
			for (int counter1 = 0; counter1 < numLabels - i - 1; counter1++) {
				indicesToRemove[counter1] = labelIndices[chain[numLabels - 1 - counter2]];
				counter2++;
			}

//			ensemble[i].initFilter(indicesToRemove, trainDataset);
			
			Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInputFormat(trainDataset);
            remove.setInvertSelection(false);
            ensemble[i].setFilter(remove);

			trainDataset.setClassIndex(labelIndices[chain[i]]);
			debug("Bulding model " + (i + 1) + "/" + numLabels);
			ensemble[i].buildClassifier(trainDataset);
		}
		
		setInference(this.inference);
		
	}

	/**
     * Learner specific implementation for predicting on specified data based on trained model.
     * This method is called from {@link #makePrediction(weka.core.Instance)} which guards for model
     * initialization and apply common handling/behavior.
     *
     * @param instance the data instance to predict on
     * @throws Exception if an error occurs while making the prediction.
     * @throws InvalidDataException if specified instance data is invalid and can not be processed by the learner
     * @return the output of the learner for the given instance
     */
	@Override
	public MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
		return inference.inferenceProcedure(instance);
	}

	/**
     * Returns a string describing the multi-label learner.
     */
	@Override
	public String toString() {
		String output = "Probabilistic Classifier Chains\n";

		StringBuffer buf = new StringBuffer();
		for (int i = 0; i < this.ensemble.length; i++)
			buf.append(this.ensemble[i].toString());

		return output + buf.toString();
	}
	
}
