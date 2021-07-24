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

		public Matrix _weight;
		public Vector _bias;

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

		// UNCOMMENT FOR DEBUGGING 
		public void SetWeight(Matrix weight)
			=> _weight = weight;
		public void SetBias(Vector bias)
			=> _bias = bias;
		

		public Vector CostGradWrtLayerBehind(Vector costGradWrtLayer, Vector layerBehind)
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

		public (Matrix, Vector) GradientDescent(Vector costGradWrtLayer, Vector layerBehind, double learningRate, out Vector costGradWrtLayerBehind)
        {
			costGradWrtLayerBehind = CostGradWrtLayerBehind(costGradWrtLayer, layerBehind);

			Matrix costGradWrtWeight = Vector.OuterProduct(costGradWrtLayer, Config.Activator(layerBehind));
			Vector costGradWrtBias = costGradWrtLayer;

			GradientDescent(costGradWrtWeight, costGradWrtBias, learningRate);
			return (costGradWrtWeight, costGradWrtBias);
		}

		public NeuralLayer DeepCopy()
			=> new(_weight.Clone(), _bias.Clone(), Config); // Config is immutable so passing it in by ref is okay


		public NeuralLayer DeepCopyWithModification(Matrix newWeight)
        {
			NeuralLayer deepCopy = DeepCopy();
			deepCopy._weight = newWeight;
			return deepCopy;
        }

		public NeuralLayer DeepCopyWithReplacement(Vector newBias)
        {
			NeuralLayer deepCopy = DeepCopy();
			deepCopy._bias = newBias;
			return deepCopy;
        }
	}
}
