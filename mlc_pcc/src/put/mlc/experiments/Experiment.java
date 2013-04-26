package put.mlc.experiments;

import java.util.ArrayList;
import java.util.List;

import put.mlc.experiments.common.ExperimentResult;
import put.mlc.experiments.common.ExperimentResults;
import put.mlc.measures.InstanceBasedFMeasure;
import put.mlc.measures.ZeroOneLossMeasure;
import mulan.classifier.MultiLabelLearner;
import mulan.evaluation.Evaluation;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroFMeasure;

/**
 * General class that allows making experiments. Every experiment class has to extend this class.
 * 
 * @author Arkadiusz Jachnik
 */
public abstract class Experiment {
	
	/**
	 * Data sets' paths and names
	 */
	protected String[] datasets = { 
			"datasets/yeast"
		};
	
	/**
	 * List of measures for the experiment
	 */
	protected List<Measure> measures = null;
	
	/**
	 * Instance of Multi-label learner (e.g. LFP, PCC, BR)
	 */
	protected MultiLabelLearner learner = null;
	
	/**
	 * if true, evaluation will be performed with multi-threaded mode of the algorithms
	 */
	protected boolean multiThreading = false;
	
	/**
	 * Default constructor with multi-label learner to set.
	 * 
	 * @param learner Multi-label learner
	 */
	public Experiment(MultiLabelLearner learner) {
		this.learner = learner;
	}

	/**
	 * Initializes measures that are desirable to evaluate.
	 * 
	 * @param numOfLabels number of labels in selected data set
	 */
	protected void initMeasures(int numOfLabels) {
		measures = new ArrayList<Measure>();
		measures.add(new HammingLoss());
		measures.add(new ZeroOneLossMeasure());
		measures.add(new InstanceBasedFMeasure());
		measures.add(new MicroFMeasure(numOfLabels));
		measures.add(new MacroFMeasure(numOfLabels));
	}
	
	/**
	 * Generates a string containing results of the evaluation.
	 * 
	 * @param ev object containing a list of evaluation measures
	 * @return a string with values of selected evaluation measures
	 */
	protected String resultsToString(Evaluation ev) {
		StringBuffer out = new StringBuffer();
		for (Measure m : ev.getMeasures()) {
			out.append(m.getName() + ": " + m.getValue() + "\n");
		}
		return out.toString();
	}
	
	/**
	 * @param datasets array of data sets' paths and names to sets
	 */
	public void setDatasets(String[] datasets) {
		this.datasets = datasets;
	}
	
	/**
	 * Runs experiment for the given train set file and test set file
	 * 
	 * @param trainSet name of ARFF file with the training set
	 * @param testSet name of ARFF file with the testing set
	 * @param labelsXML name of XML file with IDs of labels
	 * @return {@link ExperimentResult} object with results for each measure, training time and inference time
	 * @throws Exception
	 */
	public abstract ExperimentResult evaluation(String trainSet, String testSet, String labelsXML) throws Exception;
	
	/**
	 * Runs experiment for defined data sets.
	 * 
	 * @return {@link ExperimentResults} object containing {@link ExperimentResult} for each data set
	 * @throws Exception
	 */
	public ExperimentResults evaluation() throws Exception {
		ExperimentResults results = new ExperimentResults();
		for (String datasetName : this.datasets) {
			ExperimentResult result = evaluation(datasetName + "-train.arff", datasetName + "-test.arff", datasetName + ".xml");
			results.addResult(datasetName, result);
		}
		return results;
	}

	/**
	 * Runs experiment for the new set of data sets.
	 * 
	 * @param datasets array of data sets' paths and names
	 * @return {@link ExperimentResults} object containing {@link ExperimentResult} for each data set
	 * @throws Exception
	 */
	public ExperimentResults evaluation(String[] datasets) throws Exception {
		this.datasets = datasets;
		return evaluation();
	}
	
	/**
	 * @return the multiThreading
	 */
	public boolean isMultiThreading() {
		return multiThreading;
	}

	/**
	 * @param multiThreading the multiThreading to set
	 */
	public void setMultiThreading(boolean multiThreading) {
		this.multiThreading = multiThreading;
	}
	
}
