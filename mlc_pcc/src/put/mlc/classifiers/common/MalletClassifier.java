package put.mlc.classifiers.common;

import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.data.MultiLabelInstances;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.MaxEntOptimizableByLabelLikelihood;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.types.*;
import cc.mallet.util.MalletLogger;
import cc.mallet.util.MalletProgressMessageLogger;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 * Implementation of a wrapper between Mallet classifiers and Weka classifier.
 * This class extends AbstractClassifier, an abstract class from Weka
 * and runs given in constructor Mallet trainer. After training it returns
 * a Mallet classifier object.<br>
 * 
 * @author Adrian Jaroszewicz
 */
public class MalletClassifier extends AbstractClassifier {

	private static final long serialVersionUID = 5979046914468641037L;
	
	private Alphabet features;
	private LabelAlphabet labels;
	private Classifier classifier = null;
	private ClassifierTrainer<?> trainer;
	private Attribute classAttribute = null;
	
	/**
	 * Class constructor taking a ClassifierTrainer from Mallet as a parameter.
	 * 
	 * @param trainer trainer from Mallet
	 */
	public MalletClassifier(ClassifierTrainer<?> trainer) {
		this.trainer = trainer;
	}
	
	/**
	 * Default class constructor.
	 */
	public MalletClassifier() {
		this.trainer = new MaxEntTrainer();
	}
	
	/**
	 * Returns a Mallet classifier.
	 * 
	 * @return Mallet classifier
	 */
	public Classifier getClassifier() {
		return this.classifier;
	}
	
	/**
	 * Converts a single sparse instance from Weka format to Mallet format.
	 * 
	 * @param arffInstance instance in Weka format
	 * @return instance in Mallet format
	 */
	private cc.mallet.types.Instance sparseInstanceFromWekaToMallet(weka.core.Instance arffInstance) {
		int size = arffInstance.numValues();
		
		int[] sparseFeatures = new int[size];
		double[] values = new double[size];
		
		int k = 0;
		int classIndex = arffInstance.classIndex();
		for (int j = 0; j < size; j++) {
			if (classIndex != arffInstance.attributeSparse(j).index()) {
				String name = arffInstance.attributeSparse(j).name();
				sparseFeatures[k] = features.lookupIndex(name);
				values[k] = arffInstance.valueSparse(j);
				k++;
			}
		}
		FeatureVector fv = new FeatureVector(features, sparseFeatures, values);
        
		String label = arffInstance.classAttribute().value((int) arffInstance.classValue());
		return new cc.mallet.types.Instance(fv,
        		labels.lookupLabel(label),
        		"train instance", null);
	}
	
	/**
	 * Converts a single dense instance from Weka format to Mallet format.
	 * 
	 * @param arffInstance instance in Weka format
	 * @return instance in Mallet format
	 */
	private cc.mallet.types.Instance denseInstanceFromWekaToMallet(weka.core.Instance arffInstance) {
		int numberOfAttributes = arffInstance.numAttributes();
		
		int[] denseFeatures = new int[numberOfAttributes];
		double[] values = new double[numberOfAttributes];
		
		int classIndex = arffInstance.classIndex();
		for (int j = 0; j < numberOfAttributes; j++) {
			if (classIndex != arffInstance.attribute(j).index()) {
				String name = arffInstance.attribute(j).name();
				denseFeatures[j] = features.lookupIndex(name);
				values[j] = arffInstance.valueSparse(j);
			}
		}
		FeatureVector fv = new FeatureVector(features, denseFeatures, values);
        
		String label = arffInstance.classAttribute().value((int) arffInstance.classValue());
		return new cc.mallet.types.Instance(fv,
        		labels.lookupLabel(label),
        		"train instance", null);
	}
	
	/**
	 * Converts a set of instances in ARFF format to Mallet format.
	 * 
	 * @param arffInstances set of instances in ARFF format
	 * @return set of instances in Mallet format
	 */
	private InstanceList instancesFromWekaToMallet(Instances arffInstances) {
		features = new Alphabet();
		for (int i = 0; i < arffInstances.numAttributes() - 1; i++) {
			String name = arffInstances.attribute(i).name();
			features.lookupIndex(name);
		}
		
		labels = new LabelAlphabet();
		for (int i = 0; i < arffInstances.numClasses(); i++) {
			String value = arffInstances.classAttribute().value(i);
			labels.lookupIndex(value);
		}
		
		InstanceList instanceList = new InstanceList(features, labels);
		
		for (int i = 0; i < arffInstances.numInstances(); i++) {
			Instance arffInstance = arffInstances.instance(i);
	        cc.mallet.types.Instance instance = null;
	        if (arffInstance instanceof SparseInstance)
	        	instance = sparseInstanceFromWekaToMallet(arffInstance);
	        else
	        	instance = denseInstanceFromWekaToMallet(arffInstance);
	        
	        instanceList.add(instance);
	        instanceList.setInstanceWeight(instance, arffInstances.instance(i).weight());
        }
		return instanceList;
	}
	
	/**
	 * Implementation of building the model from {@link Instances} training
     * data set.
     * 
     * @param instances the training data set
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		classAttribute = instances.classAttribute().copy("");
		InstanceList instanceList = instancesFromWekaToMallet(instances);

		MalletProgressMessageLogger.getLogger(MaxEntOptimizableByLabelLikelihood.class.getName()+"-pl").setLevel(Level.OFF);
		MalletLogger.getLogger(MaxEntOptimizableByLabelLikelihood.class.getName()).setLevel(Level.OFF);
		MalletProgressMessageLogger.getLogger(MaxEntTrainer.class.getName()+"-pl").setLevel(Level.OFF);
		MalletLogger.getLogger(MaxEntTrainer.class.getName()).setLevel(Level.OFF);
		MalletLogger.getLogger("edu.umass.cs.mallet.base.ml.maximize.LimitedMemoryBFGS").setLevel(Level.OFF);
		Logger.getLogger(Classifier.class.getName()).setLevel(Level.OFF);
		MalletLogger.getLogger("global").setLevel(Level.OFF);
		MalletProgressMessageLogger.getLogger("global").setLevel(Level.OFF);
		Logger.getLogger("global").setLevel(Level.OFF);
		
		this.classifier = trainer.train(instanceList);
		trainer = null;
	}
	
	/**
	 * Returns probability distribution for the given instance.
	 * 
	 * @param wekaInstance object implementing {@link Instance}
	 */
	@Override
	public double[] distributionForInstance(Instance wekaInstance) throws Exception {
		cc.mallet.types.Instance malletInstance;
		
		if (wekaInstance instanceof SparseInstance)
			malletInstance = sparseInstanceFromWekaToMallet(wekaInstance);
		else
			malletInstance = denseInstanceFromWekaToMallet(wekaInstance);
		
		Labeling labeling = classifier.classify(malletInstance).getLabeling();

		double[] distribution = new double[labeling.numLocations()];

		for (int i = 0; i < distribution.length; i++) {
			distribution[i] = labeling.value(i);
		}

		return distribution;
	}
	
	/**
	 * Returns a class attribute.
	 * 
	 * @return class attribute
	 */
	public Attribute getClassAttribute() {
		return this.classAttribute;
	}

}
