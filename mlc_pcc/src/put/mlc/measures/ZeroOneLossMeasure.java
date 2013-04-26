package put.mlc.measures;

import mulan.evaluation.measure.LossBasedBipartitionMeasureBase;

/**
 * Implementation of the zero-one loss function.
 * 
 * @author Adrian Jaroszewicz
 */
public class ZeroOneLossMeasure extends LossBasedBipartitionMeasureBase {

	private static final long serialVersionUID = 6851931758593766341L;

	/**
     * Creates an instance of this object based on the corresponding loss
     * function.
     */
	public ZeroOneLossMeasure() {
		super(new ZeroOneLoss());
	}
}
