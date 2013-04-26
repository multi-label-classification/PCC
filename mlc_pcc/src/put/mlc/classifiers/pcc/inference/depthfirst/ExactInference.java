package put.mlc.classifiers.pcc.inference.depthfirst;

/**
 * Implementation of the inference for Probabilistic Classifier Chains.
 * It an exact version of the algorithm which is based on the Depth First 
 * Exploration.
 * 
 * @author Krzysztof Dembczynski
 */
public class ExactInference extends DepthFirstExplorationInference {
	
	private static final long serialVersionUID = -3272275001206172983L;

	/**
	 * Class constructor.
	 */
	public ExactInference() {
		setMax(0);
	}
	
	/**
	 * Returns a string containing the name of this inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public String getName() {
		return "Exact inference";
	}
}
