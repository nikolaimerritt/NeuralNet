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
        public static Parameter Zero(int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(dim => Matrix.Build.Dense(dim.row, dim.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim));

            return new Parameter(weights, biases);
        }

        public static Parameter XavierInit(params int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(dim => Math.Sqrt(6.0 / (dim.row + dim.col)) * MatrixFunctions.StdUniform(dim.row, dim.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim, 0.0));

            return new Parameter(weights, biases);
        }

        public static Parameter KaimingInit(params int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(dim => Math.Sqrt(2.0 / dim.col) * MatrixFunctions.StdNormal(dim.row, dim.col));

            var biases = BiasDims(layerSizes)
                .Select(dim => Vector.Build.Dense(dim, 0.0));

            return new Parameter(weights, biases);
        }


        public static Parameter LSUVInit(int[] layerSizes, Activation[] activators, IEnumerable<Vector> inputs, double varianceTolerance = 0.05, int maxIterations = 5)
        {
            Parameter param = GaussianOrthonormal(layerSizes);
            param.SetWeightsUnivariate(activators, inputs, varianceTolerance, maxIterations);
            return param;
        }

        public static Parameter GaussianOrthonormal(params int[] layerSizes)
        {
            var weights = WeightDims(layerSizes)
                .Select(pair => MatrixFunctions.GaussianOrthonormal(pair.Item1, pair.Item2));

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
