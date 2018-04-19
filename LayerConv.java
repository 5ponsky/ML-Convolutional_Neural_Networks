import java.util.Random;


class LayerConv extends Layer {
  int[] inputDims, filterDims, outputDims;
  int inputWidth, inputHeight, inputArea;
  int totalWeights, totalBiases, filterWidth, filterHeight, filterArea;
  int outputWidth, outputHeight, outputArea;
  Vec filter;

  int getNumberWeights() { return totalWeights; }

  /// Constructor
  LayerConv(int[] inputDims, int[] filterDims, int[] outputDims) {
    super(countTensorElements(inputDims), countTensorElements(outputDims));
    // Get the total number of tensor elements
    totalWeights = countTensorElements(filterDims);

    this.inputDims = inputDims;
    this.filterDims = filterDims;
    this.outputDims = outputDims;

    // Calculate the number of elements inside the space of the input
    outputArea = 1;
    filterArea = 1;
    for(int i = 0; i < inputDims.length; ++i) {
      filterArea *= filterDims[i];
      outputArea *= outputDims[i];
    }

    // Calculate the bias terms for the filter
    if(filterDims.length == inputDims.length) { // filter and input are same size
      totalBiases = 1;
      totalWeights += totalBiases;
    } else if(filterDims.length > inputDims.length){
      // Each tensor "plane" gets a single bias value
      totalBiases = 1;
      for(int i = inputDims.length; i < filterDims.length; ++i) {
        totalBiases *= filterDims[i];
      }
      totalWeights += totalBiases; // biases count towards weight total
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
    for(int i = 0; i < totalWeights; ++i) {
      weights.set(i, random.nextGaussian() / totalWeights);
    }
  }

  void activate(Vec weights, Vec x) {
    Tensor in = new Tensor(x, inputDims);
    Tensor out = new Tensor(activation, outputDims);

    // Strip the biases off of the weights
    Vec biases = new Vec(weights, 0, totalBiases);
    Vec filters = new Vec(weights, totalBiases, totalWeights-totalBiases);
    Tensor filter = new Tensor(filters, filterDims);

    // Call the wrapper convolution function
    Tensor.safety_convolve(in, filter, out, false);

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
    Vec filters = new Vec(weights, totalBiases, totalWeights-totalBiases);
    Tensor filter = new Tensor(filters, filterDims);

    Tensor.safety_convolve(prev_blame, filter, next_blame, true);

    return nextBlame;
  }

  void updateGradient(Vec x, Vec gradient) {

    // Splite biases and m
    Vec biases = new Vec(gradient, 0, totalBiases);
    Vec m = new Vec(gradient, totalBiases, gradient.size()-totalBiases);


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
    Tensor out = new Tensor(m, filterDims);

    System.out.println("in: " + in);
    Tensor.safety_convolve(in, kernel, out, false);
    //System.out.println("aftere: " + out); // first row of 0 after index 8
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
