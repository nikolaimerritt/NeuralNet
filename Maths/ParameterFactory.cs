using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLearning.Maths
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;
    public static class ParameterFactory
    {
        /// <summary>
        /// Returns a Paramter with weights and biases all zero.
        /// </summary>
        /// <param name="layerSizes"></param>
        /// <returns></returns>
        public static Parameter Zero(int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(dim => Matrix.Build.Dense(dim.row, dim.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim));

            return new Parameter(weights, biases);
        }

        /// <summary>
        /// Returns a Parameter with random weights and biases optimised for gradient descent using TanhSigmoid activators. Uses Xavier initialisation.
        /// </summary>
        /// <param name="layerSizes"> The layer sizes of the Parameter object. </param>
        public static Parameter XavierInit(params int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(dim => Math.Sqrt(6.0 / (dim.row + dim.col)) * MatrixFunctions.StdUniform(dim.row, dim.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim, 0.0));

            return new Parameter(weights, biases);
        }

        /// <summary>
        /// Returns a Parameter with random weights and biases optimised for gradient descent using Relu activators. Uses Kaiming-He initialisation.
        /// </summary>
        ///         /// <param name="layerSizes"> The layer sizes of the Parameter object. </param>

        public static Parameter KaimingInit(params int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(dim => Math.Sqrt(2.0 / dim.col) * MatrixFunctions.StdNormal(dim.row, dim.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim, value: 0.0));

            return new Parameter(weights, biases);
        }

        /// <summary>
        /// Returns a Parameter with random weights and biases optimised for learning the training data set with input data supplied in <paramref name="inputs"/>. Uses LSUV initialisation.
        /// </summary>
        /// <param name="inputs"> The inputs of the training data set which the Parameter will be optimised to learn.</param>
        /// <param name="activators"> The activators which the Parameter will be optimised to use. </param>
        /// <param name="varianceTolerance"> The weights will stop being adjusted when their average variance is at most an error of <paramref name="varianceTolerance"/> away from 1 </param>
        /// <param name="maxIterations"> The weights will be adjusted at most <paramref name="maxIterations"/> times. </param>
        /// <param name="layerSizes"> The layer sizes of the Parameter object. </param>
        public static Parameter LSUVInit(int[] layerSizes, Activation[] activators, IEnumerable<Vector> inputs, double varianceTolerance = 0.05, int maxIterations = 5)
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
        public static Parameter GaussianOrthonormal(params int[] layerSizes)
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
