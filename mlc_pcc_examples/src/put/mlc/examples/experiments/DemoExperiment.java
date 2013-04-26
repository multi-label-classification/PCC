package put.mlc.examples.experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import mulan.classifier.transformation.BinaryRelevance;

import put.mlc.classifiers.br.LFP;
import put.mlc.classifiers.pcc.PCC;
import put.mlc.classifiers.pcc.inference.Inference;
import put.mlc.classifiers.pcc.inference.montecarlo.FMeasureMaximizerInference;
import put.mlc.experiments.common.ExperimentResults;
import put.mlc.experiments.maxentropy.TunedExperiment;

/**
 * This class presents the experiment implementation using the classes 
 * form package put.mlc.experiments, which use MaxEntTrainer (from Mallet)
 * as an implementation of the logistic regression.
 * 
 * There are three types of experiments:
 * 1) Single Experiment - with a single learner evaluation, given a 
 *    regularization parameter.
 * 2) CV Experiment - which tunes parameters by external cross validation.
 * 3) Tuned Experiment - which tunes parameters by internal cross validation.
 * 
 * Moreover, you can create your own experiment with your MultiLabelLearner,
 * using the GeneralExperiment class.
 * 
 * @author Arkadiusz Jachnik
 */
public class DemoExperiment {
	
	public static void main(String[] args) throws FileNotFoundException {
		System.setErr(new PrintStream(new File("out.txt")));
		
		try {
			//inference algorithm for PCC
			Inference inference = new FMeasureMaximizerInference(100, 0);
			//learner object
			PCC learner = new PCC(inference);
			
			//or you can carry out an experiment with BR-based learner
			//BinaryRelevance br = null;
			//LFP learner = new LFP(br);
			
			//tuned experiment with the learner as an input
			TunedExperiment e = new TunedExperiment(learner, 3, 3, 0);
			//you can use multi-thread version of evaluation
			e.setMultiThreading(true);
			//run experiment ant get the result
			ExperimentResults res = e.evaluation();
			
			//print results to string
			System.out.println(res.toString());
		} catch(Exception ex) {
			ex.printStackTrace();
		}
	}

}
