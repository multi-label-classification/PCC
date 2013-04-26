package put.mlc.experiments.maxentropy;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import cc.mallet.classify.MaxEntTrainer;
import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import put.mlc.classifiers.common.MalletClassifier;
import put.mlc.classifiers.common.TunedClassifier;
import put.mlc.experiments.Experiment;
import put.mlc.experiments.GeneralExperiment;
import put.mlc.experiments.common.ExperimentResult;
import put.mlc.utils.MultiThreadEvaluator;
import weka.classifiers.Classifier;

/**
 * Implementation of the experiment using the Tuned Classifier (see {@link TunedClassifier},
 * which performs inner cross-validation and tunes each classifier (for each label)
 * separately.
 * 
 * @author Arkadiusz Jachnik
 * 
 * @see {@link Experiment}
 * @see {@link TunedClassifier}
 */
public class TunedExperiment extends GeneralExperiment {

	/**
	 * Regulariation parameters
	 */
	protected List<Double> variances = null;
	
	/**
	 * number of folds for inner cross-validation
	 */
	private int folds = 5;
	
	/**
	 * number of trials for inner cross-validation
	 */
	private int trials = 5; 
	
	/**
	 * seed
	 */
	private int seed = 0;
	
	/**
	 * Constructor with multi-label learner to set. Sets folds, trials and seed
	 * to default most commonly used values.
	 */
	public TunedExperiment(MultiLabelLearner learner) {
		this(learner,5,5,0);
	}
	
	/**
	 * Constructor with multi-label learner, number of folds, number of trials
	 * and seed to set.
	 * 
	 * @param learner the multi-label learner
	 * @param folds number of folds in CV
	 */
	public TunedExperiment(MultiLabelLearner learner, int folds, int trials, int seed) {
		super(learner);
		initRegulariationParameters();
		try {
			setTunedClassifierAsBaseClassifier();
		} catch (Exception e) {
			e.printStackTrace();
		}
		this.folds = folds;
		this.trials = trials; 
		this.seed = seed;
	}
	
	private void setTunedClassifierAsBaseClassifier() throws Exception {
		Classifier[] baseClassifiers = new Classifier[this.variances.size()];

		for (int i = 0; i < baseClassifiers.length; i++) {
			MaxEntTrainer maxEntTrainer = new MaxEntTrainer();
			maxEntTrainer.setGaussianPriorVariance(variances.get(i));
			
			Classifier malletClassifier = new MalletClassifier(maxEntTrainer);
			baseClassifiers[i] = malletClassifier;
		}

		TunedClassifier tc = new TunedClassifier(this.folds, this.trials, this.seed, true, baseClassifiers);

		this.setBaseCLassifierForLearner(tc);
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
}
