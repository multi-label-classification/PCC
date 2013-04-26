package put.mlc.classifiers.common;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of a classifier that tunes parameters. It chooses the best
 * of given classifiers with regard to logistic loss.<br>
 * The decision is made via internal cross validation.<br>
 *
 * @author Adrian Jaroszewicz
 */
public class TunedClassifier extends AbstractClassifier {

	private static final long serialVersionUID = 8351703638902291515L;
	
	int folds = 3;
	int trials = 3;
	int seed = 0;
	
	private boolean optimizeLogLoss = true;
	private Classifier[] baseClassifiers = null;
	protected Classifier tunedClassifier = null;

	/**
	 * Class constructor setting number of folds and trials, random seed and
	 * list of classifiers to compare.
	 * 
	 * @param folds number of folds in internal cross validation
	 * @param trials number of trials
	 * @param seed random seed
	 * @param optimizeLogLoss
	 * @param classifiers list of classifiers to compare
	 */
	public TunedClassifier(int folds, int trials, int seed, boolean optimizeLogLoss, Classifier[] classifiers) {
		this.folds = folds;
		this.trials = trials;
		this.seed = seed;
		this.optimizeLogLoss = optimizeLogLoss;
		this.baseClassifiers = classifiers;
	}
	
	/**
	 * Splits original training set for cross-validation operation.
	 * 
	 * @param dataset instances from original training set
	 * @param train set of the new training examples
	 * @param tests set of the new testing examples
	 * @param trials number of trials
	 * @param folds number of folds in CV
	 * @param random {@link Random} object
	 * @throws Exception
	 */
	private void crossData(Instances dataset, Instances[] train, Instances[] tests, int trials,
			int folds, Random random) throws Exception {
		
		Instances workingSet = new Instances(dataset);
		workingSet.randomize(random);

		for (int i = 0; i < trials; i++) {
			train[i] = workingSet.trainCV(folds, i, random);
			tests[i] = workingSet.testCV(folds, i);
		}
	}
	
	/**
	 * Tunes the parameters in cross-validation and returns the best base classifier.
	 * 
	 * @param train training examples
	 * @param folds number of folds
	 * @param trials number of trials
	 * @param seed seed value
	 * @return the best base classifier
	 * @throws Exception
	 */
	private Classifier tune(Instances train, int folds, int trials, int seed) throws Exception {
		
		//tune the parameters in cross-validation with respect to log loss or 0/1 loss
		if (this.baseClassifiers.length == 1) return baseClassifiers[0];
		
		Instances[] trains = new Instances[trials];
		Instances[] tests = new Instances[trials];
		
		crossData(train, trains, tests, trials, folds, new Random(seed));
		
		double loss = Double.MAX_VALUE;
		int best = 0;
		
		for (int j = 0; j < this.baseClassifiers.length; j++) {
			double currentLoss = test(baseClassifiers[j], trains, tests, trials);
			if (currentLoss < loss) {
				loss = currentLoss;
				best = j;
			}
		}
		
		return baseClassifiers[best];
	}

	/**
	 * Tests classifiers on the basis of a given training and testing examples.
	 * 
	 * @param classifier tested classifier
	 * @param trains training examples
	 * @param tests testing examples
	 * @param trials number of trials
	 * @return 0/1 loss
	 * @throws Exception
	 */
	private double test(Classifier classifier, Instances[] trains, Instances[] tests, int trials) throws Exception {
		double zeroOneLoss = 0;
		double logLoss = 0.0;
		int Z = 0;

		for (int i = 0; i < trials; i++) {
			Classifier testedClassifier = AbstractClassifier.makeCopy(classifier);
			
			testedClassifier.buildClassifier(trains[i]);
			
			for (int j = 0; j < tests[i].numInstances(); j++) {
				Instance instance = tests[i].instance(j);
				double[] p = testedClassifier.distributionForInstance(instance);
				
				int  y = (int) tests[i].instance(j).classValue();
				
				double tempLogLoss = 0;
				
				for(int k = 0; k < instance.numClasses(); k++) {
					tempLogLoss -= (y == k ? 1 : 0) * Math.log(p[k]);
				}
					
				
				logLoss += tempLogLoss * instance.weight();
				
				int  y_hat =  (int) testedClassifier.classifyInstance(instance);
				zeroOneLoss += (y_hat == y) ? 0 : 1;
				
				Z++;
			}
		}
		
		if (this.isOptimizeLogLoss()) return logLoss / (double) Z;
		else return zeroOneLoss / (double) Z;	
	}
	
	/**
	 * Builds tuned-classifier for original training set.
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		this.tunedClassifier = tune(instances, this.folds, this.trials, this.seed);
		this.tunedClassifier.buildClassifier(instances);
	}
	
	/**
	 * @return probability distribution for the given instance (observation).
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return this.tunedClassifier.distributionForInstance(instance);
	}
	
	/**
	 * @param optimizeLogLoss the optimizeLogLoss to set
	 */
	void setOptimizeLogLoss(boolean optimizeLogLoss) {
		this.optimizeLogLoss = optimizeLogLoss;
	}
	
	/**
	 * @return the optimizeLogLoss
	 */
	boolean isOptimizeLogLoss() {
		return optimizeLogLoss;
	}
	
}
