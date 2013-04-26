package put.mlc.measures;

import mulan.evaluation.loss.BipartitionLossFunctionBase;

/**
 * Implementation of the zero-one loss. It represents in general
 * the number of badly classified instances.
 * 
 * @author Adrian Jaroszewicz
 */
public class ZeroOneLoss extends BipartitionLossFunctionBase {

	private static final long serialVersionUID = 5589790669747115971L;

	@Override
    public String getName() {
        return "Zero/One Loss";
    }

    @Override
    public double computeLoss(boolean[] bipartition, boolean[] groundTruth) {
        for (int i = 0; i < groundTruth.length; i++)
            if (bipartition[i] != groundTruth[i])
                return 1;
        
        return 0;
    }
}