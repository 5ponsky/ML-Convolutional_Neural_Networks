import java.util.Random;


class LayerConv extends Layer {
  int[] inputDims, filterDims, outputDims;
  int filterWeights;
  Vec filter;

  int getNumberWeights() { return filterWeights; } // Unused for now?

  /// Constructor
  LayerConv(int[] inputDims, int[] filterDims, int[] outputDims) {
    super(countTensorElements(inputDims), countTensorElements(outputDims));
    filterWeights = countTensorElements(filterDims);
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
    for(int i = 0; i < filterWeights; ++i) {
      weights.set(i, random.nextGaussian() / filterWeights);
    }
  }

  void activate(Vec weights, Vec x) {
    Tensor in = new Tensor(x, inputDims);
    Tensor filter = new Tensor(weights, filterDims);
    Tensor out = new Tensor(activation, outputDims);

    Tensor.convolve(in, filter, out, false, 1);
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
