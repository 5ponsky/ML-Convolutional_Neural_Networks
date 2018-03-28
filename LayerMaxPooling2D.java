import java.util.Random;


class LayerMaxPooling2D extends Layer {


  LayerMaxPooling2D() {

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
    System.out.println("---LayerMaxPooling2D---");
    //System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    //System.out.println(activation);
    System.out.println("blame:");
    //System.out.println(blame);
  }
}
