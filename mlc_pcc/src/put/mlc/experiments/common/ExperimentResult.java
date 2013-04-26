package put.mlc.experiments.common;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import mulan.evaluation.Evaluation;
import mulan.evaluation.measure.Measure;

/**
 * Objects implementing this class contain evaluation results for each measure,
 * training time and inference time (in milliseconds). 
 * 
 * @author Arkadiusz Jachnik
 */
public class ExperimentResult {
	
	/**
	 * training time
	 */
	private long trainingTime;
	
	/**
	 * inference time
	 */
	private long testingTime;
	
	/**
	 * list of results for each measure
	 */
	private Map<String,Double> resultsForMeasures;
	
	/**
	 * decimal places for double values
	 */
	private int round = 4;
			
	/**
	 * Default constructor.
	 */
	public ExperimentResult() {
		this.trainingTime = 0;
		this.testingTime = 0;
		this.resultsForMeasures = new HashMap<String, Double>();
	}
	
	/**
	 * Constructor that fills list of results for each measure and values of training
	 * and inference time.
	 * 
	 * @param evaluation the {@link Evaluation} object containing results for each measure
	 * @param trainingTime training time in milliseconds
	 * @param testingTime testing time in milliseconds
	 */
	public ExperimentResult(Evaluation evaluation, long trainingTime, long testingTime) {
		this.resultsForMeasures = new HashMap<String, Double>();
		this.trainingTime = trainingTime;
		this.testingTime = testingTime;
		
		for (Measure measure : evaluation.getMeasures()) {
			this.resultsForMeasures.put(measure.getName(), measure.getValue());
		}
	}
	
	/**
	 * @return string with results for each measure and times
	 */
	public String toString() {
		if(this.resultsForMeasures.size() < 1) {
			return "";
		}
		
		StringBuffer out = new StringBuffer();
		for (Entry<String,Double> e : this.resultsForMeasures.entrySet()) {
			out.append(e.getKey() + " = " + numFormat(e.getValue()) + "\n");
		}
		out.append("Training time = " + this.trainingTime + "\n");
		out.append("Testing time = " + this.testingTime);
		
		return out.toString();
	}
	
	/**
	 * @return string with CSV representation of results splitting by tabs
	 */
	public String toCSVString() {
		if(this.resultsForMeasures.size() < 1) {
			return "";
		}
		
		StringBuffer out = new StringBuffer();
		for (Entry<String,Double> e : this.resultsForMeasures.entrySet()) {
			out.append(numFormat(e.getValue()) + "\t");
		}
		
		out.append(this.trainingTime + "\t" + this.testingTime);
		
		return out.toString();
	}
	
	/**
	 * @param x double value
	 * @return string with double value rounded to n (4 by default) decimal places 
	 */
	private String numFormat(double x) {
		String numFormat = "%." + this.round + "f";
		return String.format(numFormat, x);
	}
	
	/**
	 * @param round the round to set
	 */
	public void setRound(int round) {
		this.round = round;
	}

	/**
	 * @return the trainingTime
	 */
	public long getTrainingTime() {
		return trainingTime;
	}

	/**
	 * @param trainingTime the trainingTime to set
	 */
	public void setTrainingTime(long trainingTime) {
		this.trainingTime = trainingTime;
	}

	/**
	 * @return the testingTime
	 */
	public long getTestingTime() {
		return testingTime;
	}

	/**
	 * @param testingTime the testingTime to set
	 */
	public void setTestingTime(long testingTime) {
		this.testingTime = testingTime;
	}

	/**
	 * @return the resultsForMeasures
	 */
	public Map<String, Double> getResultsForMeasures() {
		return resultsForMeasures;
	}

	/**
	 * @param resultsForMeasures the resultsForMeasures to set
	 */
	public void setResultsForMeasures(Map<String, Double> resultsForMeasures) {
		this.resultsForMeasures = resultsForMeasures;
	}
	
}
