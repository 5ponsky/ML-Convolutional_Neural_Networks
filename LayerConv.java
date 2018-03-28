import java.util.Random;


class LayerConv extends Layer {

  int getNumberWeights() { return 0; }

  LayerConv() { }

  void initWeights(Vec weights, Random random) {

  }

  LayerConv(int[] inputDims, int[] filterDims, int[] outputDims) {

  }

  void activate(Vec weights, Vec x) {

  }

  Vec backProp(Vec weights, Vec prevBlame) {

  }

  void updateGradient(Vec x, Vec gradient) {

  }




  void debug() {
    System.out.println("---LayerConv---");
    //System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    //System.out.println(activation);
    System.out.println("blame:");
    //System.out.println(blame);
  }

}
