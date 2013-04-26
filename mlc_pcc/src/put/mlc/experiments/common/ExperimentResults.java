package put.mlc.experiments.common;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Set of {@link ExperimentResult} objects for each data set.
 * 
 * @author Arkadiusz Jachnik
 * 
 * @see {@link ExperimentResult}
 */
public class ExperimentResults {

	/**
	 * results stored in {@link ExperimentResult} for each data set
	 */
	private Map<String,ExperimentResult> results;
	
	/**
	 * Default constructor.
	 */
	public ExperimentResults() {
		results = new HashMap<String, ExperimentResult>();
	}
	
	/**
	 * Adds result for a given data set.
	 * 
	 * @param datasetName path and name of a given data set
	 * @param result {@link ExperimentResult} with results for each measure
	 */
	public void addResult(String datasetName, ExperimentResult result) {
		results.put(datasetName, result);
	}
	
	/**
	 * @param datasetName path and name of data set
	 * @return {@link ExperimentResult} for a given data set
	 */
	public ExperimentResult getResultForDataset(String datasetName) {
		return results.get(datasetName);
	}
	
	/**
	 * @return string array with paths and names of data sets
	 */
	public String[] getDatasets() {
		return (String[]) results.keySet().toArray();
	}
	
	/**
	 * @return string with results for each data set
	 */
	public String toString() {
		StringBuffer buf = new StringBuffer();
		for (Entry<String,ExperimentResult> entry : this.results.entrySet()) {
			buf.append(entry.getKey() + ":\n" + entry.getValue().toString());
		}
		return buf.toString();
	}
	
}
