using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Maths;

namespace NeuralNetLearning
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;
    public class Parameter
    {
        public int LayerCount
        {
            get => _weights.Length;
        }

        public int EntriesCount
        {
            get => 
                _weights.Select(w => w.RowCount * w.ColumnCount).Sum() +
                _biases.Select(b => b.Count).Sum();
        }

        private readonly Matrix[] _weights;
        private readonly Vector[] _biases;


        public Parameter(IEnumerable<Matrix> weights, IEnumerable<Vector> biases)
        {
            CheckForConstructionError(weights, biases);
            _weights = weights.ToArray();
            _biases = biases.ToArray();
        }

        public Parameter(params int[] layerSizes)
        {
            _weights = Enumerable.Range(1, layerSizes.Length - 1)
                .Select(i => MatrixFunctions.StdUniform(inputDim: layerSizes[i], outputDim: layerSizes[i-1]))
                .ToArray();

            _biases = Enumerable.Range(1, layerSizes.Length - 1)
                .Select(i => VectorFunctions.StdUniform(layerSizes[i]))
                .ToArray();
        }

        public static Parameter ReadFromDirectory(string directoryPath)
        {
            List<string> weightPaths = Directory.GetFiles(directoryPath, "weight *.csv").ToList();
            weightPaths.Sort();
            List<string> biasPaths = Directory.GetFiles(directoryPath, "bias *.csv").ToList();
            biasPaths.Sort();

            var weights = weightPaths.Select(MatrixFunctions.Read);
            var biases = biasPaths.Select(VectorFunctions.Read);

            return new Parameter(weights, biases);
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



        private static void CheckForIncompatibleOperands(Parameter lhs, Parameter rhs, string operation)
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

        private static void CheckForConstructionError(IEnumerable<Matrix> weights, IEnumerable<Vector> biases)
        {
            if (weights.Count() != biases.Count())
                throw new ArithmeticException($"Could not construct a parameters object with {weights.Count()} weights and {biases.Count()} biases, as the amount of weights must match the amount of biases.");

            foreach ((Matrix weight, Vector bias) in weights.Zip(biases))
            {
                if (weight.RowCount != bias.Count)
                    throw new ArithmeticException($"Could not construct a parameters object with a weight of dimension {(weight.RowCount, weight.ColumnCount)} and a corresponding bias of dimension {bias.Count}. The bias's dimension must match the amount of rows of the weight.");
            }
        }



        public static Parameter operator +(Parameter lhs, Parameter rhs)
        {
            CheckForIncompatibleOperands(lhs, rhs, "add");

            var newWeights = lhs._weights.Zip(rhs._weights)
                .Select(weightsPair => weightsPair.First + weightsPair.Second);

            var newBiases = lhs._biases.Zip(rhs._biases)
                .Select(biasPair => biasPair.First + biasPair.Second);

            return new Parameter(newWeights, newBiases);
        }

        public static Parameter operator *(double scalar, Parameter parameters)
        {
            var newWeights = parameters._weights
                .Select(weight => scalar * weight);

            var newBiases = parameters._biases
                .Select(bias => scalar * bias);

            return new Parameter(newWeights, newBiases);
        }

        public static Parameter operator *(Parameter lhs, Parameter rhs)
        {
            CheckForIncompatibleOperands(lhs, rhs, "multiply");

            var newWeights = lhs._weights.Zip(rhs._weights)
                .Select(weightPair => Matrix.op_DotMultiply(weightPair.First, weightPair.Second));

            var newBiases = lhs._biases.Zip(rhs._biases)
                .Select(biasPair => Vector.op_DotMultiply(biasPair.First, biasPair.Second));

            return new Parameter(newWeights, newBiases);
        }

        public static Parameter operator -(Parameter parameters)
            => (-1.0) * parameters;

        public static Parameter operator -(Parameter lhs, Parameter rhs)
        {
            CheckForIncompatibleOperands(lhs, rhs, "subtract");
            return lhs + (-rhs);
        }

        public static Parameter operator /(Parameter parameters, double scalar)
        {
            if (scalar == 0)
                throw new ArithmeticException($"Could not divide a parameters object by zero.");

            return (1 / scalar) * parameters;
        }


        public Vector[] LayerValuesBeforeActivation(Vector input, DifferentiableFunction[] activators)
        {
            if (input.Count != _biases[0].Count)
                throw new ArithmeticException($"The input has dimension {input.Count}, which does not match the required dimension of {_biases[0].Count}");

            if (LayerCount != activators.Length)
                throw new ArgumentException($"{activators.Length} activators were supplied, which does not match the amount {LayerCount} of layers of the parameters object.");

            Vector prevLayerValue = input;
            List<Vector> layerValues = new();

            for (int i = 0; i < LayerCount; i++)
            {
                Vector layerValue = _weights[i] * prevLayerValue + _biases[i];
                layerValues.Add(layerValue);
                prevLayerValue = activators[i].Apply(layerValue);
            }

            return layerValues.ToArray();
        }


        public Vector Output(Vector input, DifferentiableFunction[] activators)
            => activators.Last().Apply(LayerValuesBeforeActivation(input, activators).Last());


        public double Cost(Vector input, Vector desiredOutput, DifferentiableFunction[] activators)
            => VectorFunctions.MSE(Output(input, activators), desiredOutput);


        private Vector CostGradWrtFinalLayer(Vector output, Vector desiredOutput)
        {
            return VectorFunctions.MSEderiv(output, desiredOutput);
        }

        private Matrix CostGradWrtWeight(Vector costGradWrtLayer, Vector differentiatedLayer, Vector layerBehind)
        {
            Vector derivs = Vector.op_DotMultiply(costGradWrtLayer, differentiatedLayer);
            return Vector.OuterProduct(derivs, layerBehind);
        }


        private Vector CostGradWrtBias(Vector costGradWrtLayer, Vector differentiatedLayer)
        {
            return Vector.op_DotMultiply(costGradWrtLayer, differentiatedLayer);
        }

        private Vector CostGradWrtLayerBehind(Vector costGradWrtLayer, Vector differentiatedLayer, Matrix layerWeight)
        {
            Vector derivsFromLayer = Vector.op_DotMultiply(costGradWrtLayer, differentiatedLayer);
            return layerWeight.TransposeThisAndMultiply(derivsFromLayer);
        }

        public Parameter CostGrad(Vector input, Vector desiredOutput, DifferentiableFunction[] activators)
        {
            Vector[] layersBeforeActivation = LayerValuesBeforeActivation(input, activators);
            Vector[] layersAfterActivation = activators
                .Zip(layersBeforeActivation)
                .Select(pair => pair.First.Apply(pair.Second))
                .ToArray();

            Vector[] differeniatedLayers = activators
                .Zip(layersBeforeActivation)
                .Select(pair => pair.First.ApplyDerivative(pair.Second))
                .ToArray();

            Vector costGradWrtLayer = CostGradWrtFinalLayer(layersAfterActivation.Last(), desiredOutput);
            List<Matrix> weightCostGrads = new();
            List<Vector> biasCostGrads = new();

            for (int i = LayerCount - 1; i >= 0; i--)
            {
                Vector layerBehind = i > 0 ? layersAfterActivation[i - 1] : input;
                Matrix costGradWrtWeight = CostGradWrtWeight(costGradWrtLayer, differeniatedLayers[i], layerBehind);
                weightCostGrads.Insert(0, costGradWrtWeight);

                Vector costGradWrtBias = CostGradWrtBias(costGradWrtLayer, differeniatedLayers[i]);
                biasCostGrads.Insert(0, costGradWrtBias);

                costGradWrtLayer = CostGradWrtLayerBehind(costGradWrtLayer, differeniatedLayers[i], _weights[i]);
            }

            return new Parameter(weightCostGrads, biasCostGrads);
        }

        /*
        public bool ValueEquals(Parameter other)
        {
            if (this == other) // reference equals
                return true;

            if (LayerCount != other.LayerCount)
                return false;

            if (EntriesCount != other.EntriesCount)
                return false;

            foreach ((var weight, var otherWeight) in _weights.Zip(other._weights))
            {
                if (weight != otherWeight)
                    return false;
            }

            foreach ((var bias, var otherBias) in _biases.Zip(other._biases))
            {
                if (bias != otherBias)
                    return false;
            }

            return true;

        } */

        public double WeightEntry(int layerIdx, int row, int col)
        {
            if (LayerCount <= layerIdx)
                throw new ArgumentException($"Could not access a layer at index {layerIdx}, as there are {LayerCount} layers");

            if (_weights[layerIdx].RowCount <= row || _weights[layerIdx].ColumnCount <= col)
                throw new ArgumentException($"Could not access an entry of the weight at index {(row, col)} as the weight has dimensions {(_weights[layerIdx].RowCount, _weights[layerIdx].ColumnCount)}");

            return _weights[layerIdx][row, col];
        }

        public double BiasEntry(int layerIdx, int biasIdx)
        {
            if (LayerCount <= layerIdx)
                throw new ArgumentException($"Could not access a layer at index {layerIdx}, as there are {LayerCount} layers");

            if (_biases[layerIdx].Count <= biasIdx)
                throw new ArgumentException($"Could not access an entry of the bias at index {biasIdx} as the bias has {_biases[layerIdx].Count} entries");

            return _biases[layerIdx][biasIdx];
        }

        public Parameter DeepCopy()
        {
            var newWeights = _weights.Select(weight => weight.Clone());
            var newBiases = _biases.Select(bias => bias.Clone());
            return new Parameter(newWeights, newBiases);
        }
        /*
        public Parameters[] BasisParameters()
        {
            List<Parameters> basisParameters = new(EntriesCount);

            var zeroBiases = _biases.Select(b => 0 * b);
            for (int i = 0; i < _weights.Length; i++)
            {
                for (int r = 0; r < _weights[i].RowCount; r++)
                {
                    for (int c = 0; c < _weights[i].ColumnCount; c++)
                    {
                        Matrix[] newWeights = _weights.Select(w => 0 * w).ToArray();
                        newWeights[i][r, c] = 1;
                        basisParameters.Add(new Parameters(newWeights, zeroBiases));
                    }
                }
            }

            var zeroWeights = _weights.Select(w => 0 * w);
            for (int i = 0; i < _biases.Length; i++)
            {
                for (int c = 0; c < _biases[i].Count; c++)
                {
                    Vector[] newBiases = _biases.Select(b => 0 * b).ToArray();
                    newBiases[i][c] = 1;
                    basisParameters.Add(new Parameters(zeroWeights, newBiases));
                }
            }

            return basisParameters.ToArray();
        }*/

        public Parameter CopyWithWeight(Matrix weight, int idx)
        {
            Parameter copy = DeepCopy();
            Matrix weightToReplace = copy._weights[idx];
            if (weightToReplace.RowCount != weight.RowCount || weightToReplace.ColumnCount != weight.ColumnCount)
                throw new ArgumentException($"Could not substitute a matrix of dimension {(weightToReplace.RowCount, weightToReplace.ColumnCount)} with a matrix of dimension {(weight.RowCount, weight.ColumnCount)}");

            copy._weights[idx] = weight;
            return copy;
        }

        public Parameter CopyWithBias(Vector bias, int idx)
        {
            Parameter copy = DeepCopy();
            Vector biasToReplace = copy._biases[idx];
            if (biasToReplace.Count != bias.Count)
                throw new ArgumentException($"Could not substitute a vector of dimension {biasToReplace.Count} with a vector of dimension {bias.Count}");

            copy._biases[idx] = bias;
            return copy;
        }
    }
}
