package put.mlc.experiments.maxentropy;

import java.util.ArrayList;
import java.util.List;

import cc.mallet.classify.MaxEntTrainer;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import put.mlc.classifiers.common.MalletClassifier;
import put.mlc.experiments.Experiment;
import put.mlc.experiments.common.ExperimentResult;
import weka.classifiers.Classifier;

/**
 * Implementation of cross-validation (CV) tuning experiment.
 * 
 * @author Arkadiusz Jachnik
 * 
 * @see {@link Experiment}
 */
public class CVExperiment extends Experiment {
	
	/**
	 * number of folds in cross-validation
	 */
	private int folds = 5;
	
	/**
	 * Regulariation parameters
	 */
	protected List<Double> variances = null;
	
	private long trainingTime = 0;

	/**
	 * Constructor with multi-label learner to set.
	 * 
	 * @param learner the multi-label learner
	 */
	public CVExperiment(MultiLabelLearner learner) {
		this(learner, 5);
	}
	
	/**
	 * Constructor with multi-label learner and number of folds to set
	 * 
	 * @param learner the multi-label learner
	 * @param folds number of folds in CV
	 */
	public CVExperiment(MultiLabelLearner learner, int folds) {
		super(learner);
		initRegulariationParameters();
		this.folds = folds;
	}
	
	/**
	 * Fills a list of logistic regression regularization parameters.
	 * 
	 * @param variances list containing logistic regression regularization parameters
	 * that are going to be used in experiments
	 */
	protected void initRegulariationParameters() {
		variances = new ArrayList<Double>();
		variances.add(0.00001);
		variances.add(0.0001);
		variances.add(0.001);
		variances.add(0.01);
		variances.add(0.1);
		variances.add(1.0);
		variances.add(10.0);
		variances.add(100.0);
		variances.add(1000.0);
		variances.add(10000.0);
	}
	
	/**
	 * @param params array of logistic regression regularization parameters to set
	 */
	public void setRegulariationParameters(double[] params) {
		variances = new ArrayList<Double>();
		for (double d : params) {
			variances.add(d);
		}
	}
	
	/**
	 * Performs the cross-validation for a given regularization parameter.
	 * 
	 * @param trainSet name of ARFF file with the training set
	 * @param labelsXML name of XML file with IDs of labels
	 * @param variance the logistic regression regularization parameter
	 * @return {@link MultipleEvaluation} object with results for each measure
	 * @throws Exception
	 */
	private MultipleEvaluation crossValidation(String trainSet, String labelsXML, double variance) throws Exception {
		
		MaxEntTrainer maxEntTrainer = new MaxEntTrainer();
		maxEntTrainer.setGaussianPriorVariance(variance);

		MultiLabelInstances train = new MultiLabelInstances(trainSet, labelsXML);
		
		initMeasures(train.getNumLabels());
		
		Evaluator eval = new Evaluator();
		long trainingTimeStart = System.currentTimeMillis();
		MultipleEvaluation results = eval.crossValidate(this.learner, train, this.measures, this.folds);
		this.trainingTime += System.currentTimeMillis() - trainingTimeStart;
		
		return results;
	}

	/**
	 * Runs experiment with cross-validation tuning for each regularization parameter.
	 * Performs experiment for the best parameter end returns result.
	 * 
	 * @param trainSet name of ARFF file with the training set
	 * @param testSet name of ARFF file with the testing set
	 * @param labelsXML name of XML file with IDs of labels
	 * @return {@link ExperimentResult} object with results for each measure, training time and inference time
	 * @throws Exception
	 */
	@Override
	public ExperimentResult evaluation(String trainSet, String testSet,
			String labelsXML) throws Exception {

		double maxF = Double.MIN_VALUE;
		double bestVariance = Double.NaN;
		
		for (Double variance : this.variances) {
			MultipleEvaluation results = crossValidation(trainSet, labelsXML, variance);
			double f = results.getMean("Example-Based F-Measure");
			
			if (f > maxF) {
				maxF = f;
				bestVariance = variance;
			}
		}

		SingleExperiment se = new SingleExperiment(this.learner, bestVariance);
		
		ExperimentResult result = se.evaluation(trainSet, testSet, labelsXML);
		result.setTrainingTime(result.getTrainingTime() + this.trainingTime);
		
		return result;
	}

}
