// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.Random;
import java.lang.*;

class Main
{
	static void test(SupervisedLearner learner, String challenge) {
		// Load the training data
		String fn = "data/" + challenge;
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF(fn + "_train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF(fn + "_train_lab.arff");

		// Train the model
		//learner.train(trainFeatures, trainLabels, Training.NONE);

		// Load the test data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF(fn + "_test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF(fn + "_test_lab.arff");

		// Measure and report accuracy
		int misclassifications = learner.countMisclassifications(testFeatures, testLabels);
		System.out.println("Misclassifications by " + learner.name() + " at " + challenge + " = " + Integer.toString(misclassifications) + "/" + Integer.toString(testFeatures.rows()));
	}

	public static void testCV(SupervisedLearner learner) {
		Matrix f = new Matrix();
		f.newColumns(1);
		double[] f1 = {0};
		double[] f2 = {0};
		double[] f3 = {0};
		f.takeRow(f1);
		f.takeRow(f2);
		f.takeRow(f3);

		Matrix l = new Matrix();
		l.newColumns(1);
		double[] l1 = {2};
		double[] l2 = {4};
		double[] l3 = {6};
		l.takeRow(l1);
		l.takeRow(l2);
		l.takeRow(l3);

		double rmse = learner.cross_validation(1, 3, f, l, learner);
		System.out.println("RMSE: " + rmse);
	}

	public static void testOLS() {
		LayerLinear ll = new LayerLinear(13, 1);
		Random random = new Random(123456);
		Vec weights = new Vec(14);

		for(int i = 0; i < 14; ++i) {
			weights.set(i, random.nextGaussian());
		}

		Matrix x = new Matrix();
		x.newColumns(13);
		for(int i = 0; i < 100; ++i) {
			double[] temp = new double[13];
			for(int j = 0; j < 13; ++j) {
				temp[j] = random.nextGaussian();
			}
			x.takeRow(temp);
		}

		Matrix y = new Matrix(100, 1);
		for(int i = 0; i < y.rows(); ++i) {
			ll.activate(weights, x.row(i));
			for(int j = 0; j < ll.activation.size(); ++j) {
				double temp = ll.activation.get(j) + random.nextGaussian();
				y.row(i).set(j, temp);
			}
		}

		for(int i = 0; i < weights.size(); ++i) {
    	System.out.println(weights.get(i));
		}

		Vec olsWeights = new Vec(14);
		ll.ordinary_least_squares(x,y,olsWeights);

		System.out.println("-----------------------------");

		for(int i = 0; i < olsWeights.size(); ++i) {
			System.out.println(olsWeights.get(i));
		}
	}


	public static void testLayer() {
		double[] x = {0, 1, 2};
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		System.out.println(ll.activation.toString());
	}


	public static void opticalCharacterRecognition() {
		Random random = new Random(123456); // used for shuffling data


		/// Load training and testing data
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF("data/train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF("data/train_lab.arff");

		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF("data/test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF("data/test_lab.arff");

		/// Normalize our training/testing data by dividing by 256.0
		/// There are 256 possible values for any given entry
		trainFeatures.scale((1 / 256.0));
		testFeatures.scale((1 / 256.0));

		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainFeatures.rows()];
		int[] testIndices = new int[testFeatures.rows()];

		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }
		for(int i = 0; i < testIndices.length; ++i) { testIndices[i] = i; }

		/// Assemble and initialize a neural net
		NeuralNet nn = new NeuralNet(random);

		nn.layers.add(new LayerLinear(784, 80));
		nn.layers.add(new LayerTanh(80));

		nn.layers.add(new LayerLinear(80, 30));
		nn.layers.add(new LayerTanh(30));

		nn.layers.add(new LayerLinear(30, 10));
		nn.layers.add(new LayerTanh(10));

		nn.initWeights();


		/// Training and testing
		int mis = 10000;
		int epoch = 0;
		while(mis > 350) {
			//if(true)break;
			System.out.println("==============================");
			System.out.println("TRAINING EPOCH #" + epoch + '\n');

			mis = nn.countMisclassifications(testFeatures, testLabels);
			System.out.println("Misclassifications: " + mis);

			for(int i = 0; i < trainFeatures.rows(); ++i) {
				Vec in, target;

				// Train the network on a single input
				in = trainFeatures.row(i);

				target = new Vec(10);
				target.vals[(int) trainLabels.row(i).get(0)] = 1;

				//nn.refineWeights(in, target, nn.weights, 0.0175, Training.STOCHASTIC);
			}

			// Shuffle training and testing indices
			for(int i = 0; i < trainingIndices.length * 0.5; ++i) {
				int randomIndex = random.nextInt(trainingIndices.length);
				int temp = trainingIndices[i];
				trainingIndices[i] = trainingIndices[randomIndex];
				trainingIndices[randomIndex] = temp;

			}

			for(int i = 0; i < testIndices.length * 0.5; ++i) {
				int randomIndex = random.nextInt(testIndices.length);
				int temp = testIndices[i];
				testIndices[i] = testIndices[randomIndex];
				testIndices[randomIndex] = temp;
			}

			++epoch;
		}
	}

	public static void testBackProp() {
		double[] x = {0, 1, 2};
		Vec xx = new Vec(x);
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		double[] yhat = {9, 6};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		ll.blame = new Vec(yhat);
		ll.backProp(new Vec(m), new Vec(x));
		System.out.println(xx);
	}

	public static void testGradient() {
		double[] x = {0, 1, 2};
		Vec xx = new Vec(x);
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		Vec mm = new Vec(m);
		Vec g = new Vec(mm.size());
		g.fill(0.0);
		double[] yhat = {9, 6};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		ll.blame = new Vec(yhat);
		ll.updateGradient(xx, g);
		//System.out.println(xx);
		System.out.println(g);
	}

	public static void testNomCat() {
		Random random = new Random(123456);

		/// Load data
		Matrix data = new Matrix();
		data.loadARFF("data/hypothyroid.arff");

		// Matrix data = new Matrix(5, derp.cols());
		// data.copyBlock(0, 0, derp, 0, 0, 5, data.cols());

		/// Create a new filter to preprocess our data
		Filter f = new Filter(random);

		/// Partition the features from the labels
		Matrix features = new Matrix();
		Matrix labels = new Matrix();
		f.splitLabels(data, features, labels);

		/// PREPROCESSING
		// We need a set of preprocessors for both features and labels

		// Train the preprocessors for the training data
		f.train(features, labels, null, 0, 0.0);

		/// Partition the data into training and testing blocks
		/// With respective feature and labels blocks
		double splitRatio = 0.75;
		Matrix trainingFeatures = new Matrix();
		Matrix trainingLabels = new Matrix();
		Matrix testingFeatures = new Matrix();
		Matrix testingLabels = new Matrix();
		f.splitData(features, labels, trainingFeatures, trainingLabels,
			testingFeatures, testingLabels, 5, 0);


		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainingFeatures.rows()];
		int[] testIndices = new int[testingFeatures.rows()];

		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }
		for(int i = 0; i < testIndices.length; ++i) { testIndices[i] = i; }


		/// I want some intelligent way of getting the input and outputs
		f.nn.layers.add(new LayerLinear(trainingFeatures.cols(), 100));
		f.nn.layers.add(new LayerTanh(100));

		f.nn.layers.add(new LayerLinear(100, 4));
		f.nn.layers.add(new LayerTanh(4));

		f.nn.initWeights();

		int mis = testingLabels.rows();
		int epoch = 0;

		double testSSE = 0;
		double trainSSE = 0;

		double previous = 0;
		double tolerance = 0.0000009;


		System.out.println("batch,seconds,testRMSE,trainRMSE");
		int batch = 1;
		int batch_size = 10;
		double startTime = (double)System.nanoTime();
		while(true) {

			testSSE += f.sum_squared_error(testingFeatures, testingLabels);
			double testMSE = testSSE / testingFeatures.rows();
			double testRMSE = Math.sqrt(testMSE);

			trainSSE += f.sum_squared_error(trainingFeatures, trainingLabels);
			double trainMSE = trainSSE / trainingFeatures.rows();
			double trainRMSE = Math.sqrt(trainMSE);

			f.trainNeuralNet(trainingFeatures, trainingLabels, trainingIndices, batch_size, 0.0);
			// double mse = sse / batch;
			// double rmse = Math.sqrt(mse);

			double seconds = ((double)System.nanoTime() - startTime) / 1e9;
			System.out.println(batch + "," + seconds + "," + testRMSE + "," + trainRMSE);

			batch = batch + 1;

			// mis = f.countMisclassifications(testingFeatures, testingLabels);
			// System.out.println("mis: " + mis);

			double convergence = Math.abs(1 - (previous / testSSE));
			previous = testSSE;
			testSSE = 0;
			trainSSE = 0;
			if(convergence < tolerance) break;

		}

	}

	public static void main(String[] args)
	{
		//newTest();
		testNomCat();

	}
}
