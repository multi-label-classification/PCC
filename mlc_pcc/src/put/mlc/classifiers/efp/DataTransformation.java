package put.mlc.classifiers.efp;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.TreeSet;

import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 * This class helps in transformation of data set using the reduction 
 * scheme in EFP algorithm.
 * 
 * @author Krzysztof Dembczynski
 * @author Arkadiusz Jachnik
 */
public class DataTransformation implements Serializable {

	private static final long serialVersionUID = 7751477183655209805L;
	
	private int maxLabels = 0;
	
	private int[] labelIndices;

	int[] numRelevantLabels = null;
	
	int[][] newLabels = null;
	
	TreeSet<Integer>[] uniqueNumLabels = null; 
	
	MultiLabelInstances data = null;  
	
	Instances zeroData = null;
	
	int numberOfAllZeros = 0;

	int[] numPositives = null;
	
	public void initialize(MultiLabelInstances data) {

		maxLabels = 0;
		this.labelIndices = data.getLabelIndices();
		numRelevantLabels = new int[data.getNumInstances()];
		numPositives = new int[this.labelIndices.length];
		
		uniqueNumLabels = new TreeSet[this.labelIndices.length];
		newLabels = new int [data.getNumInstances()][this.labelIndices.length];
		
		for (int i = 0; i < this.labelIndices.length; i++) {
			uniqueNumLabels[i] = new TreeSet<Integer>();
		}
		
		for (int i = 0; i < data.getNumInstances(); i++) {
			Instance instance = data.getDataSet().instance(i);

			for (int j = 0; j < this.labelIndices.length; j++) {
				if (instance.stringValue(this.labelIndices[j]).compareTo("1") == 0) {
					numRelevantLabels[i]++;
				}
			}
		
			if (numRelevantLabels[i] > maxLabels) {
				maxLabels = numRelevantLabels[i];
			}
			
			if(numRelevantLabels[i] > 0) {
				for (int j = 0; j < this.labelIndices.length; j++) {
					if (instance.stringValue(this.labelIndices[j]).compareTo("1") == 0) {
						newLabels[i][j] = numRelevantLabels[i];
						uniqueNumLabels[j].add(newLabels[i][j]);
					}
				}
			} else {
				this.numberOfAllZeros++;
			}
		}
	}
	
	public int getNumberOfAllZeros() {
		return this.numberOfAllZeros;
	}

	public int getMaxLabels() {
		return maxLabels;
	}
	
	public Instance transformInstance(Instance instance) throws Exception {
	
		Instance transformedInstance;
		if (instance instanceof SparseInstance)
			transformedInstance = new SparseInstance(instance);
		else
			transformedInstance = new DenseInstance(instance);
		
		transformedInstance.setDataset(new Instances(data.getDataSet()));
		
		for (int j = 0; j < this.labelIndices.length; j++) {
			transformedInstance.setValue(this.labelIndices[j], "0");
		
		}

		return transformedInstance;
	}
	
	public MultiLabelInstances transformInstances(MultiLabelInstances inputData) throws Exception {

		data = inputData.clone();

		for (int i = 0; i < this.labelIndices.length; i++) {
			ArrayList<String> classValues = new ArrayList<String>(uniqueNumLabels[i].size() + 1);
			classValues.add("0");
			
			Iterator<Integer> iter = uniqueNumLabels[i].iterator();
			
			while(iter.hasNext()) {
				classValues.add(Integer.toString(iter.next()));
			}
			
			Attribute attribute = data.getDataSet().attribute(this.labelIndices[i]);
			Attribute newClass = new Attribute(attribute.name() + "_transformed", classValues);
			
			data.getDataSet().insertAttributeAt(newClass, this.labelIndices[i]);
			data.getDataSet().deleteAttributeAt(this.labelIndices[i] + 1);
		}
		
		for (int i = numRelevantLabels.length - 1; i >= 0; i--) {
			if(this.numRelevantLabels[i] > 0) {
				for (int j = 0; j < this.labelIndices.length; j++) {
					Instance instance = data.getDataSet().instance(i);
					instance.setValue(this.labelIndices[j], Integer.toString(newLabels[i][j]));
				} 
			} else {
				data.getDataSet().remove(i);
			}
		}
		
		return data;
	}
	
	public Instances transformToZeroData(MultiLabelInstances data) {
		
		this.zeroData = new Instances(data.getDataSet());
		
		// new class attribute for predicting all zeros
		ArrayList<String> classValuesForZeros = new ArrayList<String>(2);
		classValuesForZeros.add("0");
		classValuesForZeros.add("1");
		
		Attribute zeroClass = new Attribute("has_all_zeros", classValuesForZeros);
		zeroData.insertAttributeAt(zeroClass, data.getDataSet().numAttributes());
		zeroData.setClassIndex(zeroData.numAttributes() - 1);//.setClass(zeroClass);
		
		for (int i = 0; i < data.getNumInstances(); i++) {
			Instance instance = zeroData.instance(i);
			if (numRelevantLabels[i] == 0)
				instance.setValue(zeroData.numAttributes() - 1, "0");
			else
				instance.setValue(zeroData.numAttributes() - 1, "1");
		}
		
		// remove all other labels
		int[] labels = labelIndices;
		List<Integer> labelsList = new ArrayList<Integer>();
		for (int i = 0; i < labels.length; i++) {
			labelsList.add(labels[i]);
		}
		
		Collections.reverse(labelsList);
		for (int j : labelsList) {
			zeroData.deleteAttributeAt(j);
		}
		
		return this.zeroData;
	}
	
	public Instance transformToZeroInstance(Instance instance) throws Exception {
		
		double[] values = new double[this.zeroData.numAttributes()];
		double[] temp = instance.toDoubleArray();
	
		for(int i = 0; i < values.length - 1; i++) {
			values[i] = temp[i];
		}
		
		Instance transformedInstance;
		if (instance instanceof SparseInstance)
			transformedInstance = new SparseInstance(1.0, values);
		else
			transformedInstance = new DenseInstance(1.0, values);
		
		transformedInstance.setDataset(this.zeroData);
		
		return transformedInstance;
	}
}
