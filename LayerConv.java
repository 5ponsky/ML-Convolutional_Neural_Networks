import java.util.Random;


class LayerConv extends Layer {
  int[] inputDims, filterDims, outputDims;
  int filterWeights, totalBiases, filterWidth, filterHeight, filterSize;
  Vec filter;

  int getNumberWeights() { return filterWeights; }

  /// Constructor
  LayerConv(int[] inputDims, int[] filterDims, int[] outputDims) {
    super(countTensorElements(inputDims), countTensorElements(outputDims));
    this.inputDims = inputDims;
    this.filterDims = filterDims;
    this.outputDims = outputDims;

    // Since we can represent any n-tensor as endless series of
    // nxm matrices, we need to save the first 2 dimensions of the filter
    filterWidth = filterDims[0];
    filterHeight = filterDims[1];
    filterSize = filterWidth * filterHeight;

    // Get the total number of tensor elements
    filterWeights = countTensorElements(filterDims);

    // Calculate the bias terms for the filter
    if(filterDims.length < 3) { // Tensor has a depth of one
      totalBiases = 1;
      filterWeights += totalBiases;
    } else {
      // Each tensor "plane" gets a single bias value
      totalBiases = 1;
      for(int i = 2; i < filterDims.length; ++i) {
        totalBiases *= filterDims[i];
      }
      filterWeights += totalBiases;
    }
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
    Tensor out = new Tensor(activation, outputDims);

    // Strip the biases off of the weights
    Vec biases = new Vec(weights, 0, totalBiases);
    Vec filters = new Vec(weights, totalBiases, filterWeights-totalBiases);
    Tensor filter = new Tensor(filters, filterDims);

    System.out.println(filters.size());


    Tensor.convolve(in, filter, out, false, 1);

    // Vec to add bias to each tensor
    Vec bias = new Vec(filterSize);

    int pos = 0;
    for(int i = 0; i < biases.size(); ++i) {

      // Get a single filter
      Vec f = new Vec(out, pos, filterSize);
      System.out.println(f);

      // fill the bias vector with a bias value
      double b = biases.get(i);
      bias.fill(b);

      f.add(bias);

      pos += filterSize;
    }
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
