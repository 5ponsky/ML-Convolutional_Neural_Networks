import java.util.ArrayList;
import java.util.Random;

public class NeuralNet extends SupervisedLearner {
  // This number is persistent between epochs
  // It allows for decreasing learning rates
  private double learning_scale = 1.0;
  private double learning_rate = 0.0175;

  protected int trainingProgress;

  protected Vec weights;
  protected Vec gradient;
  protected ArrayList<Layer> layers;

  public int[] indices; // Bootstrapping indices

  public Vec cd_gradient;


  String name() { return ""; }

  NeuralNet(Random r) {
    super(r);
    layers = new ArrayList<Layer>();

    trainingProgress = 0;
  }

  void initWeights() {
    // Calculate the total number of weights
    int weightsSize = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      weightsSize += l.getNumberWeights();
    }
    weights = new Vec(weightsSize);
    gradient = new Vec(weightsSize);

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);

      int weightsChunk = l.getNumberWeights();
      Vec w = new Vec(weights, pos, weightsChunk);

      l.initWeights(w, this.random);

      pos += weightsChunk;
    }
  }

  Vec predict(Vec in) {
    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int weightsChunk = l.getNumberWeights();
      Vec v = new Vec(weights, pos, weightsChunk);
      l.activate(v, in);
      in = l.activation;
      pos += weightsChunk;
    }

    return (layers.get(layers.size()-1).activation);
  }

  /// Propagate blame from the output side to the input
  void backProp(Vec target) {
    Vec blame = new Vec(target.size());
    blame.add(target);
    blame.addScaled(-1, layers.get(layers.size()-1).activation);

    // keeping this around for good measure?
    layers.get(layers.size()-1).blame = new Vec(blame);

    int pos = weights.size();
    for(int i = layers.size()-1; i >= 0; --i) {
      Layer l = layers.get(i);
      //l.debug();

      int weightsChunk = l.getNumberWeights();
      pos -= weightsChunk;
      Vec w = new Vec(weights, pos, weightsChunk);

      blame = l.backProp(w, blame);
    }
  }

  void updateGradient(Vec x) {

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int gradChunk = l.getNumberWeights();
      Vec v = new Vec(gradient, pos, gradChunk);

      l.updateGradient(x, v);
      x = new Vec(l.activation);
      pos += gradChunk;
    }
  }

  /// Update the weights
  void refineWeights(double learning_rate) {
    weights.addScaled(learning_rate, gradient);
  }

  /// Trains with a set of scrambled indices to improve efficiency
  void train(Matrix features, Matrix labels, int[] indices, int batch_size, double momentum) {
    if(batch_size < 1)
      throw new IllegalArgumentException("Batch Size < 1");
    if(momentum < 0.0)
      throw new IllegalArgumentException("Momentum < 0");

    // How many patterns/mini-batches should we train on before testing?
    final int cutoff = features.rows();

    Vec in, target;
    // We want to check if we have iterated over all rows
    for(; trainingProgress < features.rows(); ++trainingProgress) {
      in = features.row(indices[trainingProgress]);
      target = labels.row(indices[trainingProgress]);

      predict(in);
      backProp(target);
      updateGradient(in);

      // Temporary
      //central_difference(in, target);

      if((trainingProgress + 1) % batch_size == 0) {
        refineWeights(learning_rate * learning_scale);
        if(momentum <= 0)
          gradient.fill(0.0);
        else
          gradient.scale(momentum);

        // Cut off for intra-training testing
        if(((trainingProgress + 1) / batch_size) % cutoff == 0) {
          ++trainingProgress;
          break;
        }
      }
    }


    // if We have trained over the entire given set
    if(trainingProgress >= features.rows()) {
      trainingProgress = 0;

      // Decrease learning rate
      if(learning_rate > 0)
        learning_scale -= 0.000001;

      scrambleIndices(random, indices, null);
    }
  }


  /// Used for estimating the gradient
  Vec central_difference(Vec x, Vec target) {
    double h = 0.00003;

    //Vec cd_gradient = new Vec(gradient.size());
    Vec validation = new Vec(weights); // Used for validating the weights

    for(int i = 0; i < weights.size(); ++i) {
      double weight = weights.get(i);

      // right side
      weights.set(i, weight + h);
      Vec right = predict(x);
      double r_res = 0.0;
      for(int j = 0; j < right.size(); ++j) {
        r_res += ((right.get(j) - target.get(j)) * (right.get(j) - target.get(j)));
      }

      // left side
      weights.set(i, weight - h);
      Vec left = predict(x);
      double l_res = 0.0;
      for(int j = 0; j < left.size(); ++j) {
        l_res += ((left.get(j) - target.get(j)) * (left.get(j) - target.get(j)));
      }

      double res = (l_res - r_res) / (2 * h);
      cd_gradient.set(i, res);

      weights.set(i, weight);

      // Validate that the weights have returned to their original values
      for(int j = 0; j < weights.size(); ++j) {
        if(weights.get(j) != validation.get(j))
          throw new RuntimeException("Error resolving weights!");
      }
    }
    return cd_gradient;
  }

}
