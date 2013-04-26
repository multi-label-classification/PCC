package put.mlc.classifiers.pcc.inference.common;

/**
 * Implementation of the binary tree, of which nodes are the {@link LabelCombination} objects.
 * 
 * @author Arkadiusz Jachnik
 * 
 * @see {@link LabelCombination}
 */
public class LabelCombinationTree {

	/**
	 * root of the binary tree
	 */
	private LabelCombination root = null;
	
	/**
	 * left child node
	 */
	private LabelCombinationTree left = null;
	
	/**
	 * right child node
	 */
	private LabelCombinationTree right = null;

	public LabelCombinationTree(LabelCombination root) {
		this.setRoot(root);
	}

	public boolean hasKids() {
		return left == null && right == null ? false : true;
	}

	public void setKids(LabelCombinationTree l, LabelCombinationTree r) {
		left = l;
		right = r;
	}

	public LabelCombinationTree left() {
		return left;
	}

	public LabelCombinationTree right() {
		return right;
	}

	public LabelCombination root() {
		return root;
	}

	public void setRoot(LabelCombination lc) {
		this.root = lc;
	}

}
