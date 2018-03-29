/// A tensor class.
class Tensor extends Vec {
	int[] dims;
	int numElements;

	/// General-purpose constructor. Example:
	/// Tensor t(v, {5, 7, 3});
	Tensor(Vec vals, int[] _dims) {
		super(vals, 0, vals.size());
		dims = new int[_dims.length];
		int tot = 1;

		for(int i = 0; i < _dims.length; i++) {
			dims[i] = _dims[i];
			tot *= _dims[i];
		}

		if(tot != vals.size())
			throw new RuntimeException("Mismatching sizes. Vec has " + Integer.toString(vals.size()) + ", Tensor has " + Integer.toString(tot));

		// Store the total number of elements
		numElements = tot;
	}

	/// Copy constructor. Copies the dimensions. Wraps the same vector.
	Tensor(Tensor copyMe) {
		super((Vec)copyMe, 0, copyMe.size());
		dims = new int[copyMe.dims.length];
		for(int i = 0; i < copyMe.dims.length; i++)
			dims[i] = copyMe.dims[i];

		numElements = copyMe.numElements;
	}

	/// The result is added to the existing contents of out. It does not replace the existing contents of out.
	/// Padding is computed as necessary to fill the the out tensor.
	/// filter is the filter to convolve with in.
	/// If flipFilter is true, then the filter is flipped in all dimensions.
	static void convolve(Tensor in, Tensor filter, Tensor out, boolean flipFilter, int stride) {
		if(out.numElements % filter.numElements != 0) {
			throw new RuntimeException("output size: " + out.numElements
				+ " / filter: " + filter.numElements + " does not have remainder 0!");
		}


		// Precompute some values
		int dc = in.dims.length;
		if(dc != filter.dims.length)
			throw new RuntimeException("Expected tensors with the same number of dimensions");
		if(dc != out.dims.length)
			throw new RuntimeException("Expected tensors with the same number of dimensions");
		int[] kinner = new int[dc];
		int[] kouter = new int[dc];
		int[] stepInner = new int[dc];
		int[] stepFilter = new int[dc];
		int[] stepOuter = new int[dc];

		// Compute step sizes
		stepInner[0] = 1;
		stepFilter[0] = 1;
		stepOuter[0] = 1;
		for(int i = 1; i < dc; i++) {
			stepInner[i] = stepInner[i - 1] * in.dims[i - 1];
			stepFilter[i] = stepFilter[i - 1] * filter.dims[i - 1];
			stepOuter[i] = stepOuter[i - 1] * out.dims[i - 1];
		}
		int filterTail = stepFilter[dc - 1] * filter.dims[dc - 1] - 1;

		// Do convolution
		int op = 0;
		int ip = 0;
		int fp = 0;
		for(int i = 0; i < dc; i++) {
			kouter[i] = 0;
			kinner[i] = 0;
			int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
			int adj = (padding - Math.min(padding, kouter[i])) - kinner[i];
			kinner[i] += adj;
			fp += adj * stepFilter[i];
		}

		// kouter
		while(true) {
			double val = 0.0;

			// Fix up the initial kinner positions
			for(int i = 0; i < dc; i++) {
				int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
				int adj = (padding - Math.min(padding, kouter[i])) - kinner[i];
				kinner[i] += adj;
				fp += adj * stepFilter[i];
				ip += adj * stepInner[i];
			}

			// kinner
			while(true) {
				val += (in.get(ip) * filter.get(flipFilter ? filterTail - fp : fp));

				// increment the kinner position
				int i;
				for(i = 0; i < dc; i++) {
					kinner[i]++;
					ip += stepInner[i];
					fp += stepFilter[i];
					int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
					if(kinner[i] < filter.dims[i] && kouter[i] + kinner[i] - padding < in.dims[i])
						break;
					int adj = (padding - Math.min(padding, kouter[i])) - kinner[i];
					kinner[i] += adj;
					fp += adj * stepFilter[i];
					ip += adj * stepInner[i];
				}
				if(i >= dc)
					break;
			}
			out.set(op, out.get(op) + val);

			// increment the kouter position
			int i;
			for(i = 0; i < dc; i++) {
				kouter[i]++;
				op += stepOuter[i];
				ip += stride * stepInner[i];
				if(kouter[i] < out.dims[i])
					break;
				op -= kouter[i] * stepOuter[i];
				ip -= kouter[i] * stride * stepInner[i];
				kouter[i] = 0;
			}
			if(i >= dc)
				break;
		}
	}

	/// Throws an exception if something is wrong.
	static void test() {

		{
			// 1D test
			Vec in = new Vec(new double[]{2,3,1,0,1});
			Tensor tin = new Tensor(in, new int[]{5});

			Vec k = new Vec(new double[]{1, 0, 2});
			Tensor tk = new Tensor(k, new int[]{3});

			Vec out = new Vec(7);
			Tensor tout = new Tensor(out, new int[]{7});

			Tensor.convolve(tin, tk, tout, true, 1);

			//     2 3 1 0 1
			// 2 0 1 --->
			Vec expected = new Vec(new double[]{2, 3, 5, 6, 3, 0, 2});
			if(Math.sqrt(out.squaredDistance(expected)) > 1e-10)
				throw new RuntimeException("wrong");
		}

		{
			// 2D test
			Vec in = new Vec(new double[] {
					1, 2, 3,
					4, 5, 6,
					7, 8, 9
			});

			Tensor tin = new Tensor(in, new int[]{3, 3});

			Vec k = new Vec(new double[] {
					1,  2,  1,
					0,  0,  0,
					-1, -2, -1
			});

			Tensor tk = new Tensor(k, new int[]{3, 3});

			Vec out = new Vec(9);
			Tensor tout = new Tensor(out, new int[]{3, 3});

			Tensor.convolve(tin, tk, tout, false, 1);

			Vec expected = new Vec(new double[] {
					-13, -20, -17,
					-18, -24, -18,
					13,  20,  17
			});

			if(Math.sqrt(out.squaredDistance(expected)) > 1e-10)
				throw new RuntimeException("wrong");
		}
	}
}
