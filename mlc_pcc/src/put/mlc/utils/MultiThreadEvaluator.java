package put.mlc.utils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instance;
import weka.core.Instances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.HierarchicalLoss;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.MacroAUC;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MeanAveragePrecision;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;

/**
 * Evaluator class based on mulan.evaluation.Evaluator. It divides the
 * prediction process into threads (by instances).
 * 
 * This is experimental beta version.
 * 
 * @author Arkadiusz Jachnik
 */
public class MultiThreadEvaluator extends Evaluator {

	// seed for reproduction of cross-validation results
	private int seed = 1;
	// number of threads
	final int NUM_OF_PROCESSORS = Runtime.getRuntime().availableProcessors();

	/**
	 * Sets the seed for reproduction of cross-validation results
	 * 
	 * @param aSeed seed for reproduction of cross-validation results
	 */
	public void setSeed(int aSeed) {
		seed = aSeed;
	}

	/**
	 * Evaluates a {@link MultiLabelLearner} on given test data set using
	 * specified evaluation measures
	 * 
	 * @param learner the learner to be evaluated via cross-validation
	 * @param data the data set for cross-validation
	 * @param measures the evaluation measures to compute
	 * @return an Evaluation object
	 * @throws IllegalArgumentException if an input parameter is null
	 * @throws Exception
	 */
	public Evaluation evaluate(MultiLabelLearner learner,
			MultiLabelInstances data, List<Measure> measures)
			throws IllegalArgumentException, Exception {
		checkLearner(learner);
		checkData(data);
		checkMeasures(measures);
		final MultiLabelInstances dataCopy = data.clone();
		final MultiLabelLearner learnerCopy = learner;
		final List<Measure> measuresCopy = measures;

		// reset measures
		for (Measure m : measures) {
			m.reset();
		}

		final int numLabels = data.getNumLabels();
		final int[] labelIndices = data.getLabelIndices();

		final Set<Measure> failed = new HashSet<Measure>();
		final Instances testData = data.getDataSet();
		int numInstances = testData.numInstances();

		ExecutorService execLoop = Executors
				.newFixedThreadPool(NUM_OF_PROCESSORS);
		final CountDownLatch latch = new CountDownLatch(numInstances);

		for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
			final int instanceIndexF = instanceIndex;
			execLoop.submit(new Runnable() {
				@Override
				public void run() {
					boolean[] trueLabels = new boolean[numLabels];
					Instance instance = testData.instance(instanceIndexF);
					if (dataCopy.hasMissingLabels(instance)) {
						return;
					}

					Instance labelsMissing = (Instance) instance.copy();
		            labelsMissing.setDataset(instance.dataset());
		            for (int i = 0; i < dataCopy.getNumLabels(); i++) {
		                labelsMissing.setMissing(dataCopy.getLabelIndices()[i]);
		            }
					
					MultiLabelOutput output = null;

					synchronized (learnerCopy) {
						try {
							output = learnerCopy.makePrediction(labelsMissing);
						} catch (InvalidDataException e) {
							e.printStackTrace();
						} catch (ModelInitializationException e) {
							e.printStackTrace();
						} catch (Exception e) {
							e.printStackTrace();
						}
					}

					synchronized (measuresCopy) {
						trueLabels = getTrueLabels(instance, numLabels, labelIndices);
						Iterator<Measure> it = measuresCopy.iterator();
						while (it.hasNext()) {
							Measure m = it.next();
							if (!failed.contains(m)) {
								try {
									m.update(output, trueLabels);
								} catch (Exception ex) {
									failed.add(m);
								}
							}
						}
					}
					
					latch.countDown();
				}
			});
		}
		
		try {
			latch.await();
		} catch (InterruptedException e2) {
			e2.printStackTrace();
		}
		execLoop.shutdown();

		return new Evaluation(measuresCopy, data);
	}

	private void checkLearner(MultiLabelLearner learner) {
		if (learner == null) {
			throw new IllegalArgumentException("Learner to be evaluated is null.");
		}
	}

	private void checkData(MultiLabelInstances data) {
		if (data == null) {
			throw new IllegalArgumentException("Evaluation data object is null.");
		}
	}

	private void checkMeasures(List<Measure> measures) {
		if (measures == null) {
			throw new IllegalArgumentException(
					"List of evaluation measures to compute is null.");
		}
	}

	private void checkFolds(int someFolds) {
		if (someFolds < 2) {
			throw new IllegalArgumentException(
					"Number of folds must be at least two or higher.");
		}
	}

	/**
	 * Evaluates a {@link MultiLabelLearner} on given test data set.
	 * 
	 * @param learner the learner to be evaluated
	 * @param data the data set for evaluation
	 * @return the evaluation result
	 * @throws IllegalArgumentException if either of input parameters is null.
	 * @throws Exception
	 */
	public Evaluation evaluate(MultiLabelLearner learner,
			MultiLabelInstances data) throws IllegalArgumentException,
			Exception {
		checkLearner(learner);
		checkData(data);

		List<Measure> measures = prepareMeasures(learner, data);

		return evaluate(learner, data, measures);
	}

	private List<Measure> prepareMeasures(MultiLabelLearner learner,
			MultiLabelInstances data) {
		List<Measure> measures = new ArrayList<Measure>();

		MultiLabelOutput prediction;
		try {
			MultiLabelLearner copyOfLearner = learner.makeCopy();
			prediction = copyOfLearner.makePrediction(data.getDataSet()
					.instance(0));
			// add bipartition-based measures if applicable
			if (prediction.hasBipartition()) {
				// add example-based measures
				measures.add(new HammingLoss());
				measures.add(new SubsetAccuracy());
				measures.add(new ExampleBasedPrecision());
				measures.add(new ExampleBasedRecall());
				measures.add(new ExampleBasedFMeasure());
				measures.add(new ExampleBasedAccuracy());
				// add label-based measures
				int numOfLabels = data.getNumLabels();
				measures.add(new MicroPrecision(numOfLabels));
				measures.add(new MicroRecall(numOfLabels));
				measures.add(new MicroFMeasure(numOfLabels));
				measures.add(new MacroPrecision(numOfLabels));
				measures.add(new MacroRecall(numOfLabels));
				measures.add(new MacroFMeasure(numOfLabels));
			}
			// add ranking-based measures if applicable
			if (prediction.hasRanking()) {
				// add ranking based measures
				measures.add(new AveragePrecision());
				measures.add(new Coverage());
				measures.add(new OneError());
				measures.add(new IsError());
				measures.add(new ErrorSetSize());
				measures.add(new RankingLoss());
			}
			// add confidence measures if applicable
			if (prediction.hasConfidences()) {
				int numOfLabels = data.getNumLabels();
				measures.add(new MeanAveragePrecision(numOfLabels));
				measures.add(new MicroAUC(numOfLabels));
				measures.add(new MacroAUC(numOfLabels));
			}
			// add hierarchical measures if applicable
			if (data.getLabelsMetaData().isHierarchy()) {
				measures.add(new HierarchicalLoss(data));
			}
		} catch (Exception ex) {
			Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null,
					ex);
		}

		return measures;
	}

	private boolean[] getTrueLabels(Instance instance, int numLabels,
			int[] labelIndices) {

		boolean[] trueLabels = new boolean[numLabels];
		for (int counter = 0; counter < numLabels; counter++) {
			int classIdx = labelIndices[counter];
			String classValue = instance.attribute(classIdx).value(
					(int) instance.value(classIdx));
			trueLabels[counter] = classValue.equals("1");
		}

		return trueLabels;
	}

	/**
	 * Evaluates a {@link MultiLabelLearner} via cross-validation on given data
	 * set with defined number of folds and seed.
	 * 
	 * @param learner the learner to be evaluated via cross-validation
	 * @param data the multi-label data set for cross-validation
	 * @param someFolds
	 * @return a {@link MultipleEvaluation} object holding the results
	 */
	public MultipleEvaluation crossValidate(MultiLabelLearner learner,
			MultiLabelInstances data, int someFolds) {
		checkLearner(learner);
		checkData(data);
		checkFolds(someFolds);

		return innerCrossValidate(learner, data, false, null, someFolds);
	}

	/**
	 * Evaluates a {@link MultiLabelLearner} via cross-validation on given data
	 * set using given evaluation measures with defined number of folds and
	 * seed.
	 * 
	 * @param learner the learner to be evaluated via cross-validation
	 * @param data the multi-label data set for cross-validation
	 * @param measures the evaluation measures to compute
	 * @param someFolds
	 * @return a {@link MultipleEvaluation} object holding the results
	 */
	public MultipleEvaluation crossValidate(MultiLabelLearner learner,
			MultiLabelInstances data, List<Measure> measures, int someFolds) {
		checkLearner(learner);
		checkData(data);
		checkMeasures(measures);

		return innerCrossValidate(learner, data, true, measures, someFolds);
	}

	private MultipleEvaluation innerCrossValidate(MultiLabelLearner learner,
			MultiLabelInstances data, boolean hasMeasures,
			List<Measure> measures, int someFolds) {
		Evaluation[] evaluation = new Evaluation[someFolds];

		Instances workingSet = new Instances(data.getDataSet());
		workingSet.randomize(new Random(seed));
		for (int i = 0; i < someFolds; i++) {
			System.out.println("Fold " + (i + 1) + "/" + someFolds);
			try {
				Instances train = workingSet.trainCV(someFolds, i);
				Instances test = workingSet.testCV(someFolds, i);
				MultiLabelInstances mlTrain = new MultiLabelInstances(train,
						data.getLabelsMetaData());
				MultiLabelInstances mlTest = new MultiLabelInstances(test,
						data.getLabelsMetaData());
				MultiLabelLearner clone = learner.makeCopy();
				clone.build(mlTrain);
				if (hasMeasures)
					evaluation[i] = evaluate(clone, mlTest, measures);
				else
					evaluation[i] = evaluate(clone, mlTest);
			} catch (Exception ex) {
				Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE,
						null, ex);
			}
		}
		MultipleEvaluation me = new MultipleEvaluation(evaluation, data);
		me.calculateStatistics();
		return me;
	}
}
