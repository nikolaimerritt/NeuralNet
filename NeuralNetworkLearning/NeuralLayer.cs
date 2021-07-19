using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Data.Text;


namespace NeuralNetLearning
{
	using Matrix = Matrix<double>;
	using Vector = Vector<double>;

	public class NeuralLayer
	{
		public int InputDim { get; private set; }
		public int OutputDim { get; private set; }
		public NeuralLayerConfig Config { get; private set; }

		private Matrix _weight;
		private Vector _bias;

        #region CONSTRUCTORS
        private NeuralLayer(Matrix weight, Vector bias, NeuralLayerConfig config)
		{
			if (weight.RowCount != bias.Count)
            {
				throw new Exception($"The weight's column count ({weight.ColumnCount}) differs from the dimension of the bias ({bias.Count})");
            }
			_weight = weight;
			_bias = bias;

			InputDim = weight.ColumnCount;
			OutputDim = weight.RowCount;
			Config = config;
		}


		public NeuralLayer(int inputDim, int outputDim, NeuralLayerConfig config) :
			this(StdNormalMatrix(inputDim, outputDim), StdNormalVector(outputDim), config) 
		{ }


		public NeuralLayer(string pathToWeight, string pathToBias, NeuralLayerConfig config) : 
			this(ReadMatrixFromFile(pathToWeight), ReadVectorFromFile(pathToBias), config) 
		{ }
        #endregion CONSTRUCTORS

        #region CONSTRUCTOR_HELPERS
        private static Matrix StdNormalMatrix(int inputDim, int outputDim)
			=> Matrix.Build.Random(outputDim, inputDim, new Normal());


		private static Vector StdNormalVector(int dim)
			=> StdNormalMatrix(1, dim).Column(0);


		private static Matrix ReadMatrixFromFile(string filepath)
			=> DelimitedReader.Read<double>(filepath, delimiter: "\t");


		private static Vector ReadVectorFromFile(string filepath)
        {
			Matrix biasAsMatrix = ReadMatrixFromFile(filepath);
			if (biasAsMatrix.RowCount == 1)
            {
				return biasAsMatrix.Row(0);
            }
			else if (biasAsMatrix.ColumnCount == 1)
            {
				return biasAsMatrix.Column(0);
            }
			else
            {
				throw new Exception($"Could not read vector from file {filepath} as there are multiple rows and multiple columns");
            }
		}

        #endregion CONSTRUCTOR_HELPERS

        public void Write(string pathToWeight, string pathToBias)
        {
			DelimitedWriter.Write(pathToWeight, _weight, delimiter: "\t");

			Matrix biasAsColMatrix = CreateMatrix.DenseOfColumnVectors(_bias);
			DelimitedWriter.Write(pathToBias, biasAsColMatrix, delimiter: "\t");
        }

		public Vector LayerValue(Vector layerBehind)
        {
			if (layerBehind.Count != _weight.ColumnCount)
            {
				throw new Exception($"The input dimension supplied to the layer ({layerBehind.Count}) differs from the input dimension of the layer ({InputDim})");
            }
			return _weight * Config.Activator(layerBehind) + _bias;
        }


		private Vector CostGradWrtLayerBehind(Vector costGradWrtLayer, Vector layerBehind)
        {
			Vector fstProduct = _weight.TransposeThisAndMultiply(costGradWrtLayer);
			Vector sndProduct = Config.ActivatorDeriv(layerBehind);
			return Vector.op_DotMultiply(fstProduct, sndProduct); // element-wise multiplication
        }

		private void GradientDescent(Matrix costGradWrtWeight, Vector costGradWrtBias, double learningRate)
        {
			_weight += (-learningRate) * costGradWrtWeight;
			_bias += (-learningRate) * costGradWrtBias;
        }

		public void GradientDescent(Vector costGradWrtLayer, Vector layerBehind, double learningRate, out Vector costGradWrtLayerBehind)
        {
			costGradWrtLayerBehind = CostGradWrtLayerBehind(costGradWrtLayer, layerBehind);

			Matrix costGradWrtWeight = Vector.OuterProduct(costGradWrtLayer, Config.Activator(layerBehind));
			Vector costGradWrtBias = costGradWrtLayer;
			GradientDescent(costGradWrtWeight, costGradWrtBias, learningRate);
		}
	}
}
