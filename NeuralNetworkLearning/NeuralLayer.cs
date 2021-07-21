using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using Maths;

namespace NeuralNetLearning
{
	using Matrix = Matrix<double>;
	using Vector = Vector<double>;
	using static MatrixFunctions;
	using static VectorFunctions;

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
			this (
				MatrixFunctions.StdUniform(inputDim, outputDim), 
				VectorFunctions.StdUniform(outputDim), 
				config
			) 
		{ }


		public NeuralLayer(string pathToWeight, string pathToBias, NeuralLayerConfig config) : 
			this (
				MatrixFunctions.Read(pathToWeight), 
				VectorFunctions.Read(pathToBias), 
				config
			) 
		{ }
        #endregion CONSTRUCTORS
		public void Write(string weightPath, string biasPath)
        {
			_weight.Write(weightPath);
			_bias.Write(biasPath);
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

		public void GradientDescent(string filepathPrefix, Vector costGradWrtLayer, Vector layerBehind, double learningRate, out Vector costGradWrtLayerBehind)
        {
			costGradWrtLayerBehind = CostGradWrtLayerBehind(costGradWrtLayer, layerBehind);

			Matrix costGradWrtWeight = Vector.OuterProduct(costGradWrtLayer, Config.Activator(layerBehind));
			Vector costGradWrtBias = costGradWrtLayer;

			string pathToWeight = $"{filepathPrefix}_weight.csv";
			string pathToBias = $"{filepathPrefix}_bias.csv";
			// Write(pathToWeight, pathToBias);

			GradientDescent(costGradWrtWeight, costGradWrtBias, learningRate);
		}
	}
}
