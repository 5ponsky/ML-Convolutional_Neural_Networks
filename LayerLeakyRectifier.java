import java.util.Random;


class LayerLeakyRectifier extends Layer {


  LayerLeakyRectifier() {

  }

  void initWeights(Vec weights, Random random) {

  }

  void activate(Vec weights, Vec x) {

  }

  Vec backProp(Vec weights, Vec prevBlame) {
    return new Vec(1);
  }

  void updateGradient(Vec x, Vec gradient) {

  }


  void debug() {
    System.out.println("---LayerLeakyRectifier---");
    //System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    //System.out.println(activation);
    System.out.println("blame:");
    //System.out.println(blame);
  }
}
