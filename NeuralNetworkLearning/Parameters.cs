using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Maths;

namespace NeuralNetworkLearning
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;
    class Parameters
    {
        public int LayerCount
        {
            get => _weights.Length;
        }
        private Matrix[] _weights;
        private Vector[] _biases;
        
        public Parameters(IEnumerable<Matrix> weights, IEnumerable<Vector> biases)
        {
            _weights = weights.ToArray();
            _biases = biases.ToArray();
        }

        public Parameters(params int[] layerSizes)
        {
            _weights = Enumerable.Range(1, layerSizes.Length)
                .Select(i => MatrixFunctions.StdUniform(inputDim: layerSizes[i], outputDim: layerSizes[i-1]))
                .ToArray();

            _biases = layerSizes
                .Select(size => VectorFunctions.StdUniform(size))
                .ToArray();
        }

        public static Parameters ReadFromDirectory(string directoryPath)
        {
            List<string> weightPaths = Directory.GetFiles(directoryPath, "weight *.csv").ToList();
            weightPaths.Sort();
            List<string> biasPaths = Directory.GetFiles(directoryPath, "bias *.csv").ToList();
            biasPaths.Sort();

            var weights = weightPaths.Select(MatrixFunctions.Read);
            var biases = biasPaths.Select(VectorFunctions.Read);

            return new Parameters(weights, biases);
        }

        public void WriteToDirectory(string directoryPath)
        {
            for (int i = 0; i < LayerCount; i++)
            {
                string weightPath = $"{directoryPath}/weight {i + 1}.csv";
                MatrixFunctions.Write(_weights[i], weightPath);
                string biasPath = $"{directoryPath}/bias {i + 1}.csv";
                VectorFunctions.Write(_biases[i], biasPath);
            }
        }

        private static void ErrorIfSizeMismatch(Parameters lhs, Parameters rhs, string operation)
        {
            if (lhs.LayerCount != rhs.LayerCount)
                throw new ArithmeticException($"Could not {operation} a parameters object with {lhs.LayerCount} layers and a parameters object with {rhs.LayerCount} layers.");

            for (int i = 0; i < lhs.LayerCount; i++)
            {
                Matrix lhsWeight = lhs._weights[i];
                Matrix rhsWeight = rhs._weights[i];

                if (lhsWeight.RowCount != rhsWeight.RowCount || lhsWeight.ColumnCount != rhsWeight.ColumnCount)
                    throw new ArithmeticException($"Could not {operation} a parameters object with weight {i} of dimension ({lhsWeight.RowCount}, {lhsWeight.ColumnCount}) and a parameters object with weight {i} of dimension ({rhsWeight.RowCount}, {rhsWeight.ColumnCount})");

                Vector lhsBias = lhs._biases[i];
                Vector rhsBias = rhs._biases[i];

                if (lhsBias.Count != rhsBias.Count)
                    throw new ArithmeticException($"Could not {operation} a parameters object with bias {i} of length {lhsBias.Count} and a parameters object with bias {i} of length {rhsBias.Count}");
            }
        }
        #region OPERATORS
        public static Parameters operator +(Parameters lhs, Parameters rhs)
        {
            ErrorIfSizeMismatch(lhs, rhs, "add");

            var newWeights = lhs._weights.Zip(rhs._weights)
                .Select(weightsPair => weightsPair.First + weightsPair.Second);

            var newBiases = lhs._biases.Zip(rhs._biases)
                .Select(biasPair => biasPair.First + biasPair.Second);

            return new Parameters(newWeights, newBiases);
        }

        public static Parameters operator *(double scalar, Parameters parameters)
        {
            var newWeights = parameters._weights
                .Select(weight => scalar * weight);

            var newBiases = parameters._biases
                .Select(bias => scalar * bias);

            return new Parameters(newWeights, newBiases);
        }

        public static Parameters operator *(Parameters lhs, Parameters rhs)
        {
            ErrorIfSizeMismatch(lhs, rhs, "multiply");

            var newWeights = lhs._weights.Zip(rhs._weights)
                .Select(weightPair => Matrix.op_DotMultiply(weightPair.First, weightPair.Second));

            var newBiases = lhs._biases.Zip(rhs._biases)
                .Select(biasPair => Vector.op_DotMultiply(biasPair.First, biasPair.Second));

            return new Parameters(newWeights, newBiases);
        }

        public static Parameters operator -(Parameters parameters)
            => (-1.0) * parameters;

        public static Parameters operator -(Parameters lhs, Parameters rhs)
        {
            ErrorIfSizeMismatch(lhs, rhs, "subtract");
            return lhs + (-rhs);
        }

        public static Parameters operator /(Parameters parameters, double scalar)
        {
            if (scalar == 0)
                throw new ArithmeticException($"Could not divide a parameters object by zero.");

            return (1 / scalar) * parameters;
        }
        #endregion OPERATORS
    }
}
