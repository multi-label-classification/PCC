package put.mlc.experiments.maxentropy;

import cc.mallet.classify.MaxEntTrainer;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import put.mlc.classifiers.common.MalletClassifier;
import put.mlc.classifiers.common.TunedClassifier;
import put.mlc.experiments.Experiment;
import put.mlc.experiments.GeneralExperiment;
import put.mlc.experiments.common.ExperimentResult;
import put.mlc.experiments.common.ExperimentResults;
import weka.classifiers.Classifier;

/**
 * Implementation of simple experiment with single evaluation of multi-label 
 * classification.
 * 
 * @author Arkadiusz Jachnik
 * 
 * @see {@link Experiment}
 */
public class SingleExperiment extends GeneralExperiment {

	/**
	 * logistic regression regularization parameter
	 */
	private double regulariationParameter = 1.0;
	
	/**
	 * Constructor with multi-label learner and regularization parameter to set
	 * 
	 * @param learner the multi-label learner
	 * @param regulariationParameter the logistic regression regularization parameter
	 */
	public SingleExperiment(MultiLabelLearner learner, double regulariationParameter) {
		super(learner);
		try {
			setMaxEntTrainerAsBaseClassifier();
		} catch (Exception e) {
			e.printStackTrace();
		}
		this.regulariationParameter = regulariationParameter;
	}
	
	private void setMaxEntTrainerAsBaseClassifier() throws Exception {
		MaxEntTrainer maxEntTrainer = new MaxEntTrainer();
		maxEntTrainer.setGaussianPriorVariance(this.regulariationParameter);
		
		Classifier malletClassifier = new MalletClassifier(maxEntTrainer);

		this.setBaseCLassifierForLearner(malletClassifier);
	}
}
