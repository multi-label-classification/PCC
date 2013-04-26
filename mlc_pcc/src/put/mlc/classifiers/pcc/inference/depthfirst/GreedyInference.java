package put.mlc.classifiers.pcc.inference.depthfirst;

/**
 * Implementation of the inference for Probabilistic Classifier Chains.
 * It an greedy version of the algorithm which is based on the Depth First 
 * Exploration.
 * 
 * @author Krzysztof Dembczynski
 */
public class GreedyInference extends DepthFirstExplorationInference {
	
	private static final long serialVersionUID = 6802190984359488961L;

	/**
	 * Class constructor.
	 */
	public GreedyInference() {
		setMax(0.5);
	}
	
	/**
	 * Returns a string containing the name of this inference method.
	 * 
	 * @return name of the inference method
	 */
	@Override
	public String getName() {
		return "Greedy inference";
	}
}
