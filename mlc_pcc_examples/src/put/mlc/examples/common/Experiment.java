package put.mlc.examples.common;

import java.util.ArrayList;
import java.util.List;

import put.mlc.measures.InstanceBasedFMeasure;
import put.mlc.measures.ZeroOneLossMeasure;

import mulan.evaluation.Evaluation;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroFMeasure;

/**
 * This class shows you how you can implement your own experiment
 * class.
 * 
 * @author Arkadiusz Jachnik
 */
public abstract class Experiment {
	protected List<String> dataSets;
	protected List<Double> regulariationParameters;
	protected List<Measure> measures;
	protected boolean isMultiThreading = true;
	
	public Experiment() {
		String[] datasets = { "datasets/yeast" };
		double[] regParams = { 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0 };
		
		initDataSetsList(datasets);
		initRegulariationParameters(regParams);
	}
	
	public void initDataSetsList(String[] datasets) {
		this.dataSets = new ArrayList<String>();
		for (String dataset : datasets) {
			this.dataSets.add(dataset);
		}
	}
	
	public void initRegulariationParameters(double[] params) {
		this.regulariationParameters = new ArrayList<Double>();
		for (double value : params) {
			this.regulariationParameters.add(value);
		}
	}
	
	public void initMeasures(int numOfLabels) {
		this.measures = new ArrayList<Measure>();
		this.measures.add(new HammingLoss());
		this.measures.add(new ZeroOneLossMeasure());
		this.measures.add(new InstanceBasedFMeasure());
		this.measures.add(new MicroFMeasure(numOfLabels));
		this.measures.add(new MacroFMeasure(numOfLabels));
	}
	
	public String resultToString(Evaluation result, long trainingTime, long testingTime) {
		StringBuffer buf = new StringBuffer();
		for (Measure measure : result.getMeasures()) {
			buf.append(measure.getName() + ": " + String.format("%.4f", measure.getValue()) + "\n");
		}
		buf.append("Training time: " + trainingTime + "ms\nTesting time: " + testingTime + "ms");
		
		return buf.toString();
	}
	
	public abstract void runExperiment() throws Exception;

}
