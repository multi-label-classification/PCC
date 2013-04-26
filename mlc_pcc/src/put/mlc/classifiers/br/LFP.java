package put.mlc.classifiers.br;

import put.mlc.classifiers.f.QuadraticNaiveFMaximizer;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * Implementation of the Label-independence-Fbeta-Plug-in-classifier 
 * algorithm (LFP).<br>
 * This algorithm is a simple binary relevance that is optimized by
 * an Adam Chai's dynamic programming algorithm (regarding the F measure).<br>
 * After building the binary relevance's model, it classifies every
 * instance from the test set and after that runs the dynamic programming
 * algorithm using given by binary relevance's model probabilities.<br>
 * For more information, see:<br>
 * <br>
 * Kian Ming Adam Chai, "Expectations of Fmeasures: Tractable Exact Computation
 * and some Empirical Observations of its Properties", SIGIR '05 Proceedings
 * of the 28th annual international ACM SIGIR conference on Research and
 * development in information retrieval, 2005.<br>
 * Pages 593-594.
 * 
 * @author Adrian Jaroszewicz
 * @author Arkadiusz Jachnik
 */
public class LFP extends MultiLabelLearnerBase {

	private static final long serialVersionUID = -6945566475355936517L;

	/**
	 * base BinaryRelevance learner
	 */
	private BinaryRelevance br;
	
	/**
	 * F-Maximizer algorithm
	 */
	private QuadraticNaiveFMaximizer FMaximizer;
	
	/**
	 * Class constructor specifying the binary relevance learner (taken from
	 * Mulan).
	 * 
	 * @param br object representing the binary relevance learner
	 * @throws Exception
	 */
	public LFP(BinaryRelevance br) throws Exception {
		this.br = br;
		this.FMaximizer = new QuadraticNaiveFMaximizer();
	}

	/**
     * Learner specific implementation of building the model from {@link MultiLabelInstances}
     * training data set. This method is called from base BinaryRelevance learner.
     *
     * @param trainingSet the training data set
     * @throws Exception if learner model was not created successfully
     */
	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		br.build(trainingSet);
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
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {
		MultiLabelOutput output;
		output = br.makePrediction(instance);
		
		double[] confidences = output.getConfidences();
		
		//boolean[] bipartition = dpfm.predictionForInstance(confidences);
		boolean[] bipartition = FMaximizer.predictionForInstance(confidences);
		MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
		return mlo;
	}
	
	/**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Kian Ming and Adam Chai");
        result.setValue(Field.TITLE, "Expectations of Fmeasures: Tractable Exact Computation and some Empirical Observations of its Properties");
        result.setValue(Field.YEAR, "2005");
        return result;
	}

	/**
     * Returns a string describing the multi-label learner.
     */
	@Override
	public String globalInfo() {
		return "Label-independence-Fbeta-Plug-in-classifier algorithm";
	}
	
	/**
	 * @param classifier the base classifier to set
	 */
	public void setBaseClassifier(Classifier classifier) {
		this.br = new BinaryRelevance(classifier);
	}
	
}
