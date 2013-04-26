package put.mlc.classifiers.pcc.inference;

import weka.core.Instance;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;

/**
 * Interface for inference procedures in PCC.
 * 
 * @author Adrian Jaroszewicz
 */
public interface IInference {

	/**
	 * Runs an inference procedure for a given instance.
	 * 
	 * @param instance instance to classify
	 * @return output of a {@link MultiLabelLearner}
	 * @throws Exception
	 */
	public MultiLabelOutput inferenceProcedure(Instance instance) throws Exception;
	
	/**
	 * Returns a string containing the name of specified inference method.
	 * 
	 * @return name of the inference method
	 */
	public String getName();
}
