using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Data.Text;


namespace NeuralNetLearning
{

	public class NeuralLayer
	{
		public int InputDim { get; private set; }
		public int OutputDim { get; private set; }
		public NeuralLayerConfig Config { get; private set; }

		private readonly Matrix<double> _weight;
		private readonly Vector<double> _bias;

        #region CONSTRUCTORS
        private NeuralLayer(Matrix<double> weight, Vector<double> bias, NeuralLayerConfig config)
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
        private static Matrix<double> StdNormalMatrix(int inputDim, int outputDim)
			=> Matrix<double>.Build.Random(outputDim, inputDim, new Normal());


		private static Vector<double> StdNormalVector(int dim)
			=> StdNormalMatrix(1, dim).Column(0);


		private static Matrix<double> ReadMatrixFromFile(string filepath)
			=> DelimitedReader.Read<double>(filepath, delimiter: "\t");


		private static Vector<double> ReadVectorFromFile(string filepath)
        {
			Matrix<double> biasAsMatrix = ReadMatrixFromFile(filepath);
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

			Matrix<double> biasAsColMatrix = CreateMatrix.DenseOfColumnVectors(_bias);
			DelimitedWriter.Write(pathToBias, biasAsColMatrix, delimiter: "\t");
        }

		public Vector<double> LayerValue(Vector<double> previousLayer)
        {
			if (previousLayer.Count != _weight.ColumnCount)
            {
				throw new Exception($"The input dimension supplied to the layer ({previousLayer.Count}) differs from the input dimension of the layer ({InputDim})");
            }
			return Config.Activator(_weight * previousLayer + _bias);
        }

		public Vector<double> CostDerivWrtLayer(Vector<double> previousLayer, Vector<double> costGradWrtPrevLayer)
        {
			Vector<double> lhs = Config.ActivatorDeriv(LayerValue(previousLayer));
			Vector<double> rhs = _weight.TransposeThisAndMultiply(costGradWrtPrevLayer);
            return Vector<double>.op_DotMultiply(lhs, rhs); // pointwise multiplication
		}
	}
}
