using System;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Data.Text;
using System.IO;

namespace CatchTheCheese
{
	public record Parameters
	{
		private Matrix<double>[] weights;
		private Vector<double>[] biases;
		public int count { get; private set; }

		public double squaredNorm
        {
			get
            {
				double weightsSquaredNorm = weights
					.Sum(weight => Math.Pow(weight.FrobeniusNorm(), 2));

				double biasesSquaredNorm = biases
					.Sum(bias => Math.Pow(bias.L2Norm(), 2));

				return weightsSquaredNorm + biasesSquaredNorm;
            }
        }

		private Parameters((Matrix<double>[] weights, Vector<double>[] biases) weightsAndBiases)
		{
			this.weights = weightsAndBiases.weights;
			this.biases = weightsAndBiases.biases;
			this.count = weightsAndBiases.weights.Length;
		}

		public Parameters(params int[] layerSizes)
		{
			(Matrix<double>[] weights, Vector<double>[] biases) = randomWeightsAndBiases(layerSizes);
			this.weights = weights;
			this.biases = biases;
			this.count = weights.Length;
		}

		public Parameters(string directoryPath) : this(readWeightsAndBiasesFromDir(directoryPath)) { }

        #region CONSTRUCTORS
        private static (Matrix<double>[], Vector<double>[]) randomWeightsAndBiases(int[] layerSizes)
		{
			List<Matrix<double>> randomWeights = new();
			List<Vector<double>> randomBiases = new();

			MatrixBuilder<double> matrixBuilder = Matrix<double>.Build;
			VectorBuilder<double> vectorBuilder = Vector<double>.Build;
			Normal standardNormal = new();

			for (int i = 0; i < layerSizes.Length - 1; i++)
			{
				int inputDim = layerSizes[i];
				int outputDim = layerSizes[i + 1];

				randomWeights.Add(matrixBuilder.Random(outputDim, inputDim, standardNormal));
				randomBiases.Add(vectorBuilder.Random(outputDim, standardNormal));
			}

			return (randomWeights.ToArray(), randomBiases.ToArray());
		}

		private static (Matrix<double>[], Vector<double>[]) readWeightsAndBiasesFromDir(string directoryPath)
		{
			List<Matrix<double>> weights = new();
			List<Vector<double>> biases = new();

			List<string> weightPaths = Directory.GetFiles(directoryPath, "weight_*.csv").ToList();
			weightPaths.Sort();
			foreach (string weightPath in weightPaths)
			{
				weights.Add(DelimitedReader.Read<double>(weightPath, delimiter: "\t"));
			}

			List<string> biasPaths = Directory.GetFiles(directoryPath, "bias_*.csv").ToList();
			biasPaths.Sort();
			foreach (string biasPath in biasPaths)
			{
				Matrix<double> vectorAsColMatrix = DelimitedReader.Read<double>(biasPath, delimiter: "\t");
				biases.Add(vectorAsColMatrix.Column(0));
			}

			return (weights.ToArray(), biases.ToArray());
		}
        #endregion CONSTRUCTORS

        public void writeToFolder(string relativePath)
		{
			for (int i = 0; i < weights.Length; i++)
			{
				DelimitedWriter.Write($"{relativePath}/weight_{i}.csv", weights[i], delimiter: "\t");
			}

			for (int i = 0; i < biases.Length; i++)
			{
				Matrix<double> vectorAsColMatrix = CreateMatrix.DenseOfColumnVectors(biases[i]);
				DelimitedWriter.Write($"{relativePath}/bias_{i}.csv", vectorAsColMatrix, delimiter: "\t");
			}
		}

        #region OPERATORS
		public static Parameters operator +(Parameters left, Parameters right)
        {
			if (left.count != right.count)
            {
				throw new ArgumentException("Cannot add two parameters of different counts");
            }

			Matrix<double>[] weights = Enumerable.Range(0, left.count)
				.Select(i => left.weights[i] + right.weights[i])
				.ToArray();

			Vector<double>[] biases = Enumerable.Range(0, left.count)
				.Select(i => left.biases[i] + right.biases[i])
				.ToArray();
			
			return new Parameters((weights, biases));
        }

		public static Parameters operator *(double scalar, Parameters parameters)
        {
			Matrix<double>[] weights = parameters.weights
				.Select(weight => scalar * weight)
				.ToArray();

			Vector<double>[] biases = parameters.biases
				.Select(bias => scalar * bias)
				.ToArray();

			return new Parameters((weights, biases));
        }

		public static Parameters operator /(Parameters parameters, double scalar)
		{
			Matrix<double>[] weights = parameters.weights
				.Select(weight => weight / scalar)
				.ToArray();

			Vector<double>[] biases = parameters.biases
				.Select(bias => bias / scalar)
				.ToArray();

			return new Parameters((weights, biases));
		}

		public static Parameters operator *(Parameters parameters, double scalar)
			=> scalar * parameters;

		public static Parameters operator -(Parameters parameters)
			=> (-1.0) * parameters;

		public static Parameters operator -(Parameters left, Parameters right)
			=> left + (-right);

        #endregion OPERATORS
    }
}
