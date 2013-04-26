package put.mlc.examples.lfp;

import java.io.File;
import java.io.PrintStream;
import java.util.List;

import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import cc.mallet.classify.MaxEntTrainer;
import put.mlc.classifiers.br.LFP;
import put.mlc.classifiers.common.MalletClassifier;
import put.mlc.classifiers.common.MultiThreadTunedClassifier;
import put.mlc.classifiers.common.TunedClassifier;
import put.mlc.examples.common.Experiment;
import put.mlc.measures.InstanceBasedFMeasure;
import put.mlc.utils.MultiThreadEvaluator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;

/**
 * This class shows you how you can implement your own tuned-based
 * experiment class.
 * 
 * @author Arkadiusz Jachnik
 */
public class TunedExperiment extends Experiment {
	
	private int folds;
	private int trials;
	private int seed;
	private long trainingTime = 0;
	private long testingTime = 0;

	public TunedExperiment() {
		super();
		this.folds = 5;
		this.trials = 5;
		this.seed = 0;
	}
	
	public TunedExperiment(int folds, int trials, int seed) {
		super();
		this.folds = folds;
		this.trials = trials;
		this.seed = seed;
	}

	private Evaluation tunedExperiment(String dataset) throws Exception {

		MultiLabelInstances trainSet = new MultiLabelInstances(dataset + "-train.arff", dataset + ".xml");
		MultiLabelInstances testSet = new MultiLabelInstances(dataset + "-test.arff", dataset + ".xml");
		
		Classifier[] baseClassifiers = new Classifier[this.regulariationParameters.size()];
		
		for (int i = 0; i < baseClassifiers.length; i++) {
			MaxEntTrainer maxEntTrainer = new MaxEntTrainer();
			maxEntTrainer.setGaussianPriorVariance(this.regulariationParameters.get(i));
			
			Classifier malletClassifier = new MalletClassifier(maxEntTrainer);
			baseClassifiers[i] = malletClassifier;
		}
		
		AbstractClassifier tunedClassifier = isMultiThreading ?
				new TunedClassifier(this.folds, this.trials, this.seed, true, baseClassifiers) :
					new MultiThreadTunedClassifier(this.folds, this.trials, this.seed, true, baseClassifiers);
		
		BinaryRelevance BRBaseLearner = new BinaryRelevance(tunedClassifier);
		LFP LFPLearner = new LFP(BRBaseLearner);
		
		long trainingTimeStart = System.currentTimeMillis();
		LFPLearner.build(trainSet);
		this.trainingTime = System.currentTimeMillis() - trainingTimeStart;
		
		this.initMeasures(trainSet.getNumLabels());
		
		Evaluator eval = this.isMultiThreading ? new MultiThreadEvaluator() : new Evaluator();
		long testingTimeStart = System.currentTimeMillis();
		Evaluation results = eval.evaluate(LFPLearner, testSet, this.measures);
		this.testingTime = System.currentTimeMillis() - testingTimeStart;

		return results;
	}
	
	@Override
	public void runExperiment() throws Exception {
		for (String dataset : this.dataSets) {
			System.out.println("Experiment for \"" + dataset + "\":");
			
			Evaluation results = tunedExperiment(dataset);
			System.out.println(this.resultToString(results, this.trainingTime, this.testingTime) + "\n");
		}
	}

	public static void main(String[] args) throws Exception {
		System.setErr(new PrintStream(new File("errors.txt")));
		Experiment experiment = new TunedExperiment(3,3,0);
		experiment.runExperiment();
	}
}
