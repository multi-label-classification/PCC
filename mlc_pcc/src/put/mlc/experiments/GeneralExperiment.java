package put.mlc.experiments;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import cc.mallet.classify.MaxEntTrainer;
import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import put.mlc.classifiers.common.MalletClassifier;
import put.mlc.classifiers.common.TunedClassifier;
import put.mlc.experiments.common.ExperimentResult;
import put.mlc.utils.MultiThreadEvaluator;
import weka.classifiers.Classifier;

/**
 * Implementation of the simple experiment when the learner with a base classifier is given.
 * 
 * @author Arkadiusz Jachnik
 * 
 * @see {@link Experiment}
 */
public class GeneralExperiment extends Experiment {
	
	/**
	 * Constructor with multi-label learner to set. 
	 */
	public GeneralExperiment(MultiLabelLearner learner) {
		super(learner);
	}
	
	protected void setBaseCLassifierForLearner(Classifier baseClassifier) throws Exception {
		Method setBaseClassifier = null;
		try {
			setBaseClassifier = this.learner.getClass().getMethod("setBaseClassifier", Classifier.class);
		} catch (SecurityException e) {
		  e.printStackTrace();
		  throw new Exception(e.getMessage());
		} catch (NoSuchMethodException e) {
		  e.printStackTrace();
		  throw new Exception(e.getMessage());
		}
		
		try {
			setBaseClassifier.invoke(this.learner, baseClassifier);
		} catch (IllegalArgumentException e) {
			e.printStackTrace();
			throw new Exception(e.getMessage());
		} catch (IllegalAccessException e) {
			e.printStackTrace();
			throw new Exception(e.getMessage());
		} catch (InvocationTargetException e) {
			e.printStackTrace();
			throw new Exception(e.getMessage());
		}
	}
	
	/**
	 * Runs experiment using a given multi-label learner.
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
		MultiLabelInstances train = new MultiLabelInstances(trainSet, labelsXML);
		MultiLabelInstances test = new MultiLabelInstances(testSet, labelsXML);
	
		long trainingTimeStart = System.currentTimeMillis();
		this.learner.build(train);
		long trainingTime = System.currentTimeMillis() - trainingTimeStart;
		
		initMeasures(train.getNumLabels());
		
		Evaluator eval = isMultiThreading() ? new MultiThreadEvaluator() : new Evaluator();
		long testingTimeStart = System.currentTimeMillis();
		Evaluation results = eval.evaluate(this.learner, test, measures);
		long testingTime = System.currentTimeMillis() - testingTimeStart;
		
		return new ExperimentResult(results, trainingTime, testingTime);
	}

}
