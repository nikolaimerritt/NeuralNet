using MathNet.Numerics.LinearAlgebra;
using NeuralNetLearning.Maths.Activations;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetLearning.Maths
{
    using Matrix = Matrix<double>;
    using Vector = Vector<double>;
    internal static class ParameterFactory
    {
        /// <summary>
        /// Returns a Paramter with weights and biases all zero.
        /// </summary>
        /// <param name="layerSizes"></param>
        /// <returns></returns>
        internal static Parameter Zero(int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(dim => Matrix.Build.Dense(dim.row, dim.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim));

            return new Parameter(weights, biases);
        }

        /// <summary>
        /// Reads the <see cref="Parameter"/> that has been written to <paramref name="directoryPath"/> using <see cref="Parameter.WriteToDirectory(string)"/>.
        /// <para>
        /// The returned <see cref="Parameter"/> has equivalent weight and bias values compared to the written <see cref="Parameter"/>.
        /// </para>
        /// </summary>
        /// <param name="directoryPath">The (relative or absolute) path to the directory storing the Parameter.</param>
        internal static Parameter ReadFromDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                throw new DirectoryNotFoundException($"Could not find directory {directoryPath}");

            if (!Directory.EnumerateFiles(directoryPath).Any())
                return null;

            List<string> weightPaths = Directory.GetFiles(directoryPath, "weight *.csv").ToList();
            weightPaths.Sort();
            List<string> biasPaths = Directory.GetFiles(directoryPath, "bias *.csv").ToList();
            biasPaths.Sort();

            var weights = weightPaths.Select(MatrixFunctions.Read);
            var biases = biasPaths.Select(VectorFunctions.Read);

            return new Parameter(weights, biases);
        }

        /// <summary>
        /// Returns a <see cref="Parameter"/> initialised with random weights and biases that are optimised for the use of <see cref="TanhActivation"/>. 
        /// <para>
        /// Uses <see href="https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf">Xavier initialisation</see>.
        /// </para>
        /// </summary>
        /// <param name="layerSizes"> The layer sizes of the Parameter object. </param>
        internal static Parameter XavierInit(params int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(dim => Math.Sqrt(6.0 / (dim.row + dim.col)) * MatrixFunctions.StdUniform(dim.row, dim.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim, 0.0));

            return new Parameter(weights, biases);
        }

        /// <summary>
        /// Returns a <see cref="Parameter"/> initialised with random weights and biases that are optimised for the use of <see cref="ReluActivation"/>. 
        /// <para>
        /// Uses <see href="https://arxiv.org/abs/1502.01852v1">Kaiming He</see> initialisation.
        /// </para>
        /// </summary>
        /// <param name="layerSizes"> The layer sizes of the Parameter object. </param>
        internal static Parameter KaimingInit(params int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(dim => Math.Sqrt(2.0 / dim.col) * MatrixFunctions.StdNormal(dim.row, dim.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim, value: 0.0));

            return new Parameter(weights, biases);
        }

        /// <summary>
        /// Returns a <see cref="Parameter"/> with random weights and biases optimised for learning the training data set with input data supplied in <paramref name="inputs"/>.
        /// <para>
        /// Uses <see href="http://cmp.felk.cvut.cz/~mishkdmy/papers/mishkin-iclr2016.pdf">LSUV initialisation</see>.
        /// </para>
        /// </summary>
        /// <param name="inputs"> The inputs of the training data set which the Parameter will be optimised to learn.</param>
        /// <param name="activators"> The activators which the Parameter will be optimised to use. </param>
        /// <param name="varianceTolerance"> The weights will stop being adjusted when their average variance is at most an error of <paramref name="varianceTolerance"/> away from 1 </param>
        /// <param name="maxIterations"> The weights will be adjusted at most <paramref name="maxIterations"/> times. </param>
        /// <param name="layerSizes"> The layer sizes of the Parameter object. </param>
        internal static Parameter LSUVInit(int[] layerSizes, Activation[] activators, IEnumerable<Vector> inputs, double varianceTolerance = 0.05, int maxIterations = 5)
        {
            Parameter param = GaussianOrthonormal(layerSizes);
            param.SetWeightsUnivariate(activators, inputs, varianceTolerance, maxIterations);
            return param;
        }

        /// <summary>
        /// Returns a Parameter object with zero biases, and orthonormal weights constructed from a Gaussian matrix.
        /// </summary>
        /// <param name="layerSizes"> The layer sizes of the Parameter object </param>
        /// <returns></returns>
        internal static Parameter GaussianOrthonormal(params int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(pair => MatrixFunctions.GaussianOrthonormal(pair.row, pair.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim, 0));

            return new Parameter(weights, biases);
        }

        private static List<(int row, int col)> WeightDims(int[] layerSizes)
            => Enumerable
            .Range(0, layerSizes.Length - 1)
            .Select(i => (layerSizes[i + 1], layerSizes[i]))
            .ToList();

        private static List<int> BiasDims(int[] layerSizes)
            => layerSizes
            .ToList()
            .GetRange(1, layerSizes.Length - 1);
    }
}
