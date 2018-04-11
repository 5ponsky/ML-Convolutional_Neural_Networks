import java.util.Random;


class LayerConv extends Layer {
  int[] inputDims, filterDims, outputDims;
  int inputWidth, inputHeight, inputArea;
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

    // Calculating important properties now for input (input)
    inputWidth = inputDims[0];
    inputHeight = inputDims[1];
    inputArea = inputWidth * inputHeight;

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

    // Vec to add bias to each output tensor
    Vec bias = new Vec(outputArea);

    int biasPos = 0;
    int outputPos = 0;
    for(int i = 0; i < biases.size(); ++i) {
      // fill the bias vector with a bias value
      double b = biases.get(i);
      bias.fill(b);

      // Get a 2-tensor from the output
      Vec o = new Vec(out, outputPos, outputArea);

      // add the bias value to the corrseponding 2-tensor
      o.add(bias);

      ++biasPos;
      outputPos += outputArea;
    }
  }

  Vec backProp(Vec weights, Vec prevBlame) {
    blame.fill(0.0);
    blame.add(prevBlame);

    Tensor prev_blame = new Tensor(prevBlame, outputDims);

    Vec nextBlame = new Vec(inputs);
    Tensor next_blame = new Tensor(nextBlame, inputDims);

    Vec biases = new Vec(weights, 0, totalBiases); // ignore b
    Vec filters = new Vec(weights, totalBiases, filterWeights-totalBiases);
    Tensor filter = new Tensor(filters, filterDims);

    int pbPos = 0;
    int filterPos = 0;
    for(int i = filter.extra_dimensions()-1; i >=0 ; --i) {
      // Wrap a prevBlame vector
      Vec v = new Vec(prevBlame, pbPos, outputArea);
      int[] reducedPBDims = prev_blame.reduced_dimensions();
      Tensor pb = new Tensor(v, reducedPBDims);

      // Wrap a filter vector
      Vec w = new Vec(filter, filterPos, filterArea);
      int[] reducedFDims = filter.reduced_dimensions();
      Tensor f = new Tensor(w, reducedFDims);

      Tensor.convolve(pb, f, next_blame, true, 1);

    }
    return nextBlame;
  }

  void updateGradient(Vec x, Vec gradient) {
    System.out.println("x: " + x.size());
    System.out.println("blame: " + blame.size());
    System.out.println("totalB: " + totalBiases);

    Vec biases = new Vec(gradient, 0, totalBiases);

    int pos = 0;
    for(int i = 0; i < biases.size(); ++i) {
      Vec v = new Vec(blame, pos, outputArea);
      biases.set(i, biases.get(i) + v.innerSum());
      pos += outputArea;
    }

    // Wrap in the input/activation from previous layer
    Tensor in = new Tensor(x, inputDims);

    // Wrap the blame output (kernel in this case)
    Tensor kernel = new Tensor(blame, outputDims);

    // In this case the filter/gradient is the output
    Tensor out = new Tensor(gradient, filterDims);

    in.printDims();
    kernel.printDims();
    out.printDims();
    //Tensor.convolve

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
