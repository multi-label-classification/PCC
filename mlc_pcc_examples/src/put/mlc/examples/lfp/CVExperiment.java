package put.mlc.examples.lfp;

import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import cc.mallet.classify.MaxEntTrainer;
import put.mlc.classifiers.br.LFP;
import put.mlc.classifiers.common.MalletClassifier;
import put.mlc.examples.common.Experiment;
import put.mlc.measures.InstanceBasedFMeasure;
import put.mlc.utils.MultiThreadEvaluator;
import weka.classifiers.Classifier;

/**
 * This class shows you how you can implement your own cross-validation-based
 * experiment class.
 * 
 * @author Arkadiusz Jachnik
 */
public class CVExperiment extends Experiment {
	
	private int folds;
	private long trainingTime = 0;
	private long testingTime = 0;

	public CVExperiment() {
		super();
		this.folds = 5;
	}
	
	public CVExperiment(int folds) {
		super();
		this.folds = folds;
	}

	private MultipleEvaluation crossValidation(String dataset, double variance) throws Exception {
		
		MultiLabelInstances trainSet = new MultiLabelInstances(dataset + "-train.arff", dataset + ".xml");
		
		MaxEntTrainer maxEntTrainer = new MaxEntTrainer();
		maxEntTrainer.setGaussianPriorVariance(variance);
		Classifier malletClassifier = new MalletClassifier(maxEntTrainer);
		
		BinaryRelevance BRBaseLearner = new BinaryRelevance(malletClassifier);

		LFP LFPLearner = new LFP(BRBaseLearner);
		
		this.initMeasures(trainSet.getNumLabels());
		
		Evaluator eval = this.isMultiThreading ? new MultiThreadEvaluator() : new Evaluator();
		
		long trainingTimeStart = System.currentTimeMillis();
		MultipleEvaluation results = eval.crossValidate(LFPLearner, trainSet, this.measures, this.folds);
		this.trainingTime += System.currentTimeMillis() - trainingTimeStart;
		
		return results;
	}
	
	private Evaluation singleEvaluation(String dataset, double variance) throws Exception {
		
		MultiLabelInstances trainSet = new MultiLabelInstances(dataset + "-train.arff", dataset + ".xml");
		MultiLabelInstances testSet = new MultiLabelInstances(dataset + "-test.arff", dataset + ".xml");
		
		MaxEntTrainer maxEntTrainer = new MaxEntTrainer();
		maxEntTrainer.setGaussianPriorVariance(variance);
		Classifier malletClassifier = new MalletClassifier(maxEntTrainer);
		
		BinaryRelevance BRBaseLearner = new BinaryRelevance(malletClassifier);
		LFP LFPLearner = new LFP(BRBaseLearner);
		
		long trainingTimeStart = System.currentTimeMillis();
		LFPLearner.build(trainSet);
		this.trainingTime += System.currentTimeMillis() - trainingTimeStart;
		
		this.initMeasures(trainSet.getNumLabels());
		
		Evaluator eval = this.isMultiThreading ? new MultiThreadEvaluator() : new Evaluator();
		long testingTimeStart = System.currentTimeMillis();
		Evaluation results = eval.evaluate(LFPLearner, testSet, this.measures);
		this.testingTime += System.currentTimeMillis() - testingTimeStart;
		
		return results;
	}
	
	@Override
	public void runExperiment() throws Exception {
		for (String dataset : this.dataSets) {
			System.out.println("Experiment for \"" + dataset + "\":");
			
			double maxF = Double.MIN_VALUE;
			double bestVariance = Double.NaN;
			
			for (double variance : this.regulariationParameters) {	
				MultipleEvaluation me = crossValidation(dataset, variance);
				double f = me.getMean(InstanceBasedFMeasure.measureName);
				if (f > maxF) {
					maxF = f;
					bestVariance = variance;
				}
			}
			
			Evaluation results = singleEvaluation(dataset, bestVariance);
			System.out.println(this.resultToString(results, this.trainingTime, this.testingTime) + "\n");
		}
	}

	public static void main(String[] args) throws Exception {
		//System.setErr(new PrintStream(new File("errors.txt")));
		Experiment experiment = new CVExperiment(3);
		experiment.runExperiment();
	}
}
