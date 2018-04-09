import java.util.Random;


class LayerMaxPooling2D extends Layer {
  static final int[] pooling_dims = {2, 2}; // Emulating a nxn matrix as a vector
  static final int poolsize = 4; // multiply all elements of the pool together

  Matrix pooling, maxMap; // Pool matrix for pooling operation
  int width, depth, height, planeSize;

  int getNumberWeights() { return 0; }

  LayerMaxPooling2D(int width, int height, int depth) {
    super(width * height, (width * height * depth) / poolsize);
    this.width = width;
    this.height = height;
    this.depth = depth;
    this.planeSize = width * height;

    maxMap = new Matrix((height * depth) / poolsize, width / poolsize);

    pooling = new Matrix(pooling_dims);

    // Error checking to make sure we can pool over the planar dimensions
    if(width % poolsize != 0)
      throw new IllegalArgumentException("W: " + width + " / " + poolsize + " not an integer");
    if(height % poolsize != 0)
      throw new IllegalArgumentException("H: " + height + " / " + poolsize + " not an integer");
  }

  /// Pool over each 2-tensor
  void activate(Vec weights, Vec x) {
    if(x.size() % poolsize != 0)
      throw new RuntimeException("input vector cannot be rastered evenly");

    // Push the input vector into a matrix
    int x_pos = 0;
    Matrix input = new Matrix();
    input.setSize(height * depth, width); // extend the height of the matrix for depth
    for(int i = 0; i < input.rows(); ++i) {
      for(int j = 0; j < input.cols(); ++j) {
        input.row(i).set(j, x.get(x_pos));
        ++x_pos;
      }
    }

    // Pool over the input vector
    int pos = 0;
    for(int i = 0; i < pooling.rows() * depth; ++i) { // pool over the whole depth of each matrix
      for(int j = 0; j < pooling.cols(); ++j) {
        pooling.copyBlock(0, 0, input, i * pooling.rows(), j * pooling.cols(), pooling.rows(), pooling.cols());

        // TODO: mark each value that was the max for backprop

        // find the max value
        double max = pooling.maxValue();
        activation.set(pos, max);
        ++pos;
      }
    }

  }

  Vec backProp(Vec weights, Vec prevBlame) {
    // Whichever value happened to be the maximum, it carries the blame
    return new Vec(1);
  }

  void updateGradient(Vec x, Vec gradient) {
  } // LayerMaxPooling2D contains no weights so this is empty

  void initWeights(Vec weights, Random random) {
  } // LayerMaxPooling2D contains no weights so this is empty

  void debug() {
    System.out.println("---LayerMaxPooling2D---");
    System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    System.out.println(activation);
    System.out.println("blame:");
    System.out.println(blame);
  }
}
