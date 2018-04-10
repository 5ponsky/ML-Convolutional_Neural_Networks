import java.util.Random;


class LayerMaxPooling2D extends Layer {
  static final int[] pooling_dims = {2, 2}; // Emulating a nxn matrix as a vector
  static final int poolsize = 4; // multiply all elements of the pool together

  //Vec maxMap; // Saves the index of the maxmimum value for backProp
  boolean[][] maxMap;

  Matrix pooling; // Pool matrix for pooling operation
  int width, depth, height, planeSize;

  int getNumberWeights() { return 0; }

  LayerMaxPooling2D(int width, int height, int depth) {
    super(width * height * depth, (width * height * depth) / poolsize);
    this.width = width;
    this.height = height;
    this.depth = depth;
    this.planeSize = width * height;

    //maxMap = new Vec(width*height*depth);
    maxMap = new boolean[height * depth][width];
    //maxMap = new Matrix((height * depth) / pooling_dims[0], width / pooling_dims[1]);
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
    int index = 0; // tracks where we are in the matrix
    int row = 0;
    int col = 0;
    for(int i = 0; i < pooling.rows() * depth; ++i) { // pool over the whole depth of each matrix
      for(int j = 0; j < pooling.cols(); ++j) {
        pooling.copyBlock(0, 0, input, i * pooling.rows(), j * pooling.cols(), pooling.rows(), pooling.cols());
        row = 0;
        col = 0;

        // find the max value and retain its index
        double max = pooling.row(0).get(0);
        int maxIndex = 0;

        for(int k = 0; k < pooling.rows(); ++k) {
          for(int l = 0; l < pooling.cols(); ++l) {

            if(pooling.row(k).get(l) > max) {
              max = pooling.row(k).get(l);
              maxIndex = index;
              row = k;
              col = l;
            }
            ++index;
          }
        }


        // Save the max value and its index
        maxMap[i * pooling.rows() + row][j * pooling.cols() + col] = true;
        activation.set(pos, max);
        ++pos;
      }
    }

  }

  Vec backProp(Vec weights, Vec prevBlame) {

    Vec nextBlame = new Vec(inputs);

    int pos = 0;
    int pb = 0;
    for(int i = 0; i < maxMap.length; ++i) {

      for(int j = 0; j < maxMap[0].length; ++j) {
        if(maxMap[i][j] == true) {
          double blame = prevBlame.get(pb);
          nextBlame.set(pos, blame);
          ++pb;
        }
        else
          nextBlame.set(pos, 0.0);

        ++pos;
      }
    }

    return nextBlame;
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
