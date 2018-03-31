import java.util.Random;


class LayerMaxPooling2D extends Layer {
  static final int poolsize = 4; // Emulating a nxn matrix as a vector
  int width, depth, height, planeSize;

  int getNumberWeights() { return 0; }

  LayerMaxPooling2D(int width, int height, int depth) {
    super(width * height, (width * height) / poolsize);
    this.width = width;
    this.height = height;
    this.depth = depth;
    this.planeSize = width * height;

    // Error checking to make sure we can pool over the planar dimensions
    if(width % poolsize != 0)
      throw new IllegalArgumentException("W: " + width + " / " + poolsize + " not an integer");
    if(height % poolsize != 0)
      throw new IllegalArgumentException("H: " + height + " / " + poolsize + " not an integer");
  }

  void activate(Vec weights, Vec x) {
    if(x.size() % poolsize != 0)
      throw new RuntimeException("input vector cannot be rastered evenly");

    int pos = 0;
    for(int i = 0; i < depth; ++i) {
      for(int j = 0; j < activation.size(); ++j) {
        Vec v = new Vec(x, pos, poolsize);
        double max = v.findMax();
        activation.set(j, max);
        pos += poolsize;
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
