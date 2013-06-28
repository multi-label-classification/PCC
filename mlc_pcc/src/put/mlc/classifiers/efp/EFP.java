package put.mlc.classifiers.efp;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import put.mlc.classifiers.f.FMeasure;

import weka.classifiers.functions.Logistic;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Implementation of Exact-F-Plug-in classifier (EFP) algorithm.<br>
 * 
 * Another algorithm from binary relevance family. It is optimized by the
 * General F-measure Maximizer. For more information, see:<br>
 * <br>
 * Krzysztof Dembczynski, Arkadiusz Jachnik, Wojciech Kotlowski, Willem Waegeman, 
 * Eyke Hullermeier, "Optimizing the F-Measure in Multi-Label Classification: Plug-in 
 * Rule Approach versus Structured Loss Minimization",
 * In: Proceedings of the International Conference on Machine Learning, 2013.<br>
 * 
 * @author Krzysztof Dembczynski
 * @author Arkadiusz Jachnik
 */
public class EFP extends TransformationBasedMultiLabelLearner {

	private static final long serialVersionUID = -8081673809038221406L;

	/**
	 * transforms data set using the reduction scheme
	 */
	private DataTransformation dataTransformation = null;

	/**
	 * transformed data set
	 */
	private MultiLabelInstances transformedData = null;
	
	/**
	 * the ensemble of binary relevance models
	 */
	protected FilteredClassifier[] ensemble;
	
	/**
	 * classifier for prediction consisting of all 0s
	 */
	protected Classifier allZeros;

	/**
	 * Default constructor. It is used only with tuned experiment.
	 */
	public EFP() {
		//do something...
	}
	
	/**
	 * Class constructor setting a base classifier for binary relevance.
	 * 
	 * @param classifier the base classifier which will be used internally to handle the data
     * @see Classifier
	 */
	public EFP(Classifier classifier) {
		super(classifier);
	}

	/**
     * Learner specific implementation of building the model from {@link MultiLabelInstances}
     * training data set.
     *
     * @param train the training data set
     * @throws Exception if learner model was not created successfully
     */
	protected void buildInternal(MultiLabelInstances input) throws Exception {
        
		labelIndices = input.getLabelIndices();
		numLabels = input.getNumLabels();
		ensemble = new FilteredClassifier[numLabels];
		dataTransformation = new DataTransformation();
		dataTransformation.initialize(input);
		
		if(dataTransformation.getNumberOfAllZeros() > 0) {
			Instances trainAllZero = dataTransformation.transformToZeroData(input);
			
			debug("Bulding model for all zeros");
			
			allZeros = AbstractClassifier.makeCopy(baseClassifier);
			allZeros.buildClassifier(trainAllZero);
		}
		
		transformedData = dataTransformation.transformInstances(input);
		Instances trainingData = transformedData.getDataSet();
		
		for (int i = 0; i < numLabels; i++) {
			ensemble[i] = new FilteredClassifier();
			ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

			// Indices of attributes to remove
			int[] indicesToRemove = new int[numLabels - 1];
			int counter2 = 0;
			for (int j = 0; j < numLabels; j++) {
				if (labelIndices[j] != labelIndices[i]) {
					indicesToRemove[counter2] = labelIndices[j];
					counter2++;
				}
			}

			Remove remove = new Remove();
			remove.setAttributeIndicesArray(indicesToRemove);
			remove.setInputFormat(trainingData);
			remove.setInvertSelection(false);
			ensemble[i].setFilter(remove);

			trainingData.setClassIndex(labelIndices[i]);

			debug("Bulding model " + (i + 1) + "/" + numLabels);
			
			ensemble[i].buildClassifier(trainingData);
		}
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
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {

		double p_0 = 0.0;
		
		if(dataTransformation.getNumberOfAllZeros() > 0) {
			Instance transformedZeroInstance = dataTransformation.transformToZeroInstance(instance);
			p_0 = allZeros.distributionForInstance(transformedZeroInstance)[transformedZeroInstance.classAttribute().indexOfValue("0")];
		}
		
		Instance transformedInstance = dataTransformation.transformInstance(instance);
		
		double[][] probabilities = new double[dataTransformation.getMaxLabels()][numLabels];
		double[] marginals = new double[numLabels];
		
		for (int i = 0; i < numLabels; i++) {
			double[] distribution = ensemble[i].distributionForInstance(transformedInstance);
			for (int j = 0; j < distribution.length; j++) {
				int label = Integer.parseInt(transformedInstance.attribute(this.labelIndices[i]).value(j));
				if(label != 0) {
					probabilities[label - 1][i] = (1 - p_0) * distribution[j];
					marginals[i] = (1 - p_0) * distribution[j];
				}
			}
		}
		
		FMeasure f = new FMeasure();
		f.initialize(numLabels, dataTransformation.getMaxLabels(), probabilities, p_0);		
		f.computeFMeasureMaximizer();
		
		MultiLabelOutput prediction = f.computePrediction();
		
		return prediction;
	}
	
	/**
	 * Sets the base classifier.
	 * @param baseClassifier base classifier
	 */
	public void setBaseClassifier(Classifier baseClassifier) {
		this.baseClassifier = baseClassifier;
	}

}