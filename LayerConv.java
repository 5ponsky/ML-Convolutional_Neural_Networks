import java.util.Random;


class LayerConv extends Layer {
  int[] inputDims, filterDims, outputDims;
  Vec filter;

  int getNumberWeights() { return 0; } // Unused for now?

  /// Constructor
  LayerConv(int[] inputDims, int[] filterDims, int[] outputDims) {
    super(countTensorElements(inputDims), countTensorElements(outputDims));
    this.inputDims = inputDims;
    this.filterDims = filterDims;
    this.outputDims = outputDims;
  }

  /// Counts the number of inputs and outputs
  private static int countTensorElements(int[] dims) {
    int n = 1;
    for(int i = 0; i < dims.length; ++i) {
      n *= dims[i];
    }
    return n;
  }

  /// Initialize the filter weights
  void initWeights(Vec weights, Random random) {
    int totalElements = countTensorElements(filterDims);

    for(int i = 0; i < totalElements; ++i) {
      weights.set(i, random.nextGaussian() / totalElements);
    }
  }

  void activate(Vec weights, Vec x) {

  }

  Vec backProp(Vec weights, Vec prevBlame) {
    return new Vec(1);
  }

  void updateGradient(Vec x, Vec gradient) {

  }




  void debug() {
    System.out.println("---LayerConv---");
    //System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    System.out.println(activation);
    System.out.println("blame:");
    System.out.println(blame);
  }

}
