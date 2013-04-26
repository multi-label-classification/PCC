package put.mlc.measures;

import mulan.evaluation.measure.ExampleBasedBipartitionMeasureBase;

/**
 * This class calculates example based F-measure, correction of the
 * mulan.evaluation.measure.ExampleBasedFMeasure class.
 * 
 * @author Adrian Jaroszewicz
 */
public class InstanceBasedFMeasure extends ExampleBasedBipartitionMeasureBase {

	private static final long serialVersionUID = -267641156033420969L;
	private double beta = 1.0;
	public static final String measureName = "Example-Based F-Measure";

	@Override
    protected void updateBipartition(boolean[] prediction, boolean[] truth) {
		int sumPrediction = 0;
		int sumTruth = 0;
		int sumMul = 0;
		for (int i = 0; i < truth.length; i++) {
			if (prediction[i])
				sumPrediction++;
			if (truth[i])
				sumTruth++;
			int b = prediction[i]? 1 : 0;
			int t = truth[i]? 1 : 0;
			sumMul += b * t;
		}
		
		if (sumPrediction == 0 && sumTruth == 0)
			sum += 1;
		else
			sum += ((1 + beta * beta) * sumMul) / (sumPrediction + beta * beta * sumTruth);
		count++;
    }

	@Override
	public double getIdealValue() {
		return 1;
	}

	@Override
	public String getName() {
		return InstanceBasedFMeasure.measureName;
	}
	
}