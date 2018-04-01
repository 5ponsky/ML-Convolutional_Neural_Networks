import java.util.Random;


class LayerConv extends Layer {
  int[] inputDims, filterDims, outputDims;
  int filterWeights, totalBiases, filterWidth, filterHeight, filterArea;
  int outputWidth, outputHeight, outputArea;
  Vec filter;

  int getNumberWeights() { return filterWeights; }

  /// Constructor
  LayerConv(int[] inputDims, int[] filterDims, int[] outputDims) {
    super(countTensorElements(inputDims), countTensorElements(outputDims));
    this.inputDims = inputDims;
    this.filterDims = filterDims;
    this.outputDims = outputDims;

    // Calculating important properties now to save computation (filter)
    filterWidth = filterDims[0];
    filterHeight = filterDims[1];
    filterArea = filterWidth * filterHeight;

    // Calculating output properties for use with filter
    outputWidth = outputDims[0];
    outputHeight = outputDims[1];
    outputArea = outputWidth * outputHeight;

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

    // Call the wrapper convolution function
    Tensor.convolve(in, filter, out, false);

    // TODO: add functionality to break down inputs into 2-tensors for safety also

    // TODO: make a wrapper for tensors

    //Tensor.convolve(in, filter, out, false, 1);

    // int pos = 0;
    // int[] tempDims = new int[2];
    // for(int i = 0; i < totalBiases; ++i) { // totalBiases in this context is # of filter planes
    //
    //   // wrap a single filter as a vector
    //   Vec tempFilter = new Vec(filters, pos, filterArea);
    //
    //   // wrap a 2-tensor from the output
    //   Vec tempOut = new Vec(x, )
    //
    //   // reduce the filterDims to length 2
    //   tempDims[0] = filterWidth;
    //   tempDims[1] = filterHeight;
    //
    //   // wrap it as a tensor
    //   Tensor t = new Tensor(temp, tempDims);
    //
    //   Tensor.convolve(in, t, out, false, 1);
    // }

    // Vec to add bias to each tensor
    // Vec bias = new Vec(filterArea);
    //
    // pos = 0; // reset pos for next operation
    // for(int i = 0; i < biases.size(); ++i) {
    //
    //   // Get a single filter
    //   Vec f = new Vec(out, pos, filterArea);
    //   System.out.println(f);
    //
    //   // get a single output frame
    //   // Vec o = new Vec(out, pos, )
    //
    //   // fill the bias vector with a bias value
    //   double b = biases.get(i);
    //   bias.fill(b);
    //
    //   // add the bias value to the corrseponding 2-tensor
    //   f.add(bias);
    //
    //   pos += filterArea;
    // }
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
