using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.Statistics;
using Maths;

namespace NeuralNetLearning.Maths
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;
    public class Parameter
    {
        public int LayerCount
        {
            get => _weights.Length;
        }

        public int[] LayerSizes
        {
            get 
            {
                List<int> layerSizes = new() { _weights.First().ColumnCount };
                layerSizes.AddRange(_biases.Select(b => b.Count));
                return layerSizes.ToArray();
            }
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

        public static Parameter ReadFromDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                throw new DirectoryNotFoundException($"Could not find directory {directoryPath}");

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
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            for (int i = 0; i < LayerCount; i++)
            {
                string weightPath = $"{directoryPath}/weight {i + 1}.csv";
                _weights[i].Write(weightPath);
                string biasPath = $"{directoryPath}/bias {i + 1}.csv";
                _biases[i].Write(biasPath);
            }
        }

        private static void CheckForIncompatibleOperands(Parameter lhs, Parameter rhs, string operation)
        {
            if (lhs.LayerCount != rhs.LayerCount)
                throw new ArithmeticException($"Could not {operation} a parameter object with {lhs.LayerCount} layers and a parameter object with {rhs.LayerCount} layers.");

            for (int i = 0; i < lhs.LayerCount; i++)
            {
                Matrix lhsWeight = lhs._weights[i];
                Matrix rhsWeight = rhs._weights[i];

                if (lhsWeight.RowCount != rhsWeight.RowCount || lhsWeight.ColumnCount != rhsWeight.ColumnCount)
                    throw new ArithmeticException($"Could not {operation} a parameter object with weight {i} of dimension ({lhsWeight.RowCount}, {lhsWeight.ColumnCount}) and a parameter object with weight {i} of dimension ({rhsWeight.RowCount}, {rhsWeight.ColumnCount})");

                Vector lhsBias = lhs._biases[i];
                Vector rhsBias = rhs._biases[i];

                if (lhsBias.Count != rhsBias.Count)
                    throw new ArithmeticException($"Could not {operation} a parameter object with bias {i} of length {lhsBias.Count} and a parameter object with bias {i} of length {rhsBias.Count}");
            }
        }

        private static void CheckForConstructionError(IEnumerable<Matrix> weights, IEnumerable<Vector> biases)
        {
            if (weights.Count() != biases.Count())
                throw new ArithmeticException($"Could not construct a parameter object with {weights.Count()} weights and {biases.Count()} biases, as the amount of weights must match the amount of biases.");

            foreach ((Matrix weight, Vector bias) in weights.Zip(biases))
            {
                if (weight.RowCount != bias.Count)
                    throw new ArithmeticException($"Could not construct a parameter object with a weight of dimension {(weight.RowCount, weight.ColumnCount)} and a corresponding bias of dimension {bias.Count}. The bias's dimension must match the amount of rows of the weight.");
            }
        }

        public void InPlaceAdd(Parameter other)
        {
            CheckForIncompatibleOperands(this, other, "add");
            for (int l = 0; l < LayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] += other._weights[l][r, c];
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] += other._biases[l][i];
                }
            }
        }

        public void InPlaceMultiply(double scalar)
        {
            for (int l = 0; l < LayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] *= scalar;
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] *= scalar;
                }
            }
        }

        public void InPlaceDivide(double scalar)
        {
            if (scalar == 0)
                throw new ArithmeticException($"Could not divide a parameter object by zero.");

            for (int l = 0; l < LayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] /= scalar;
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] /= scalar;
                }
            }
        }

        public static Parameter operator +(Parameter left, Parameter right)
        {
            Parameter result = left.DeepCopy();
            result.InPlaceAdd(right);
            return result;
        }

        public static Parameter operator *(double scalar, Parameter parameter)
        {
            Parameter result = parameter.DeepCopy();
            result.InPlaceMultiply(scalar);
            return result;
        }

        public static Parameter operator *(Parameter lhs, Parameter rhs)
        {
            CheckForIncompatibleOperands(lhs, rhs, "multiply");

            var newWeights = lhs._weights.Zip(rhs._weights, Matrix.op_DotMultiply);
            var newBiases = lhs._biases.Zip(rhs._biases, Vector.op_DotMultiply);

            return new Parameter(newWeights, newBiases);
        }

        public static Parameter operator -(Parameter parameter)
            => (-1.0) * parameter;

        public static Parameter operator -(Parameter lhs, Parameter rhs)
        {
            CheckForIncompatibleOperands(lhs, rhs, "subtract");
            return lhs + (-rhs);
        }

        public static Parameter operator /(Parameter parameter, double scalar)
        {
            Parameter result = parameter.DeepCopy();
            result.InPlaceDivide(scalar);
            return result;
        }


        public static Parameter operator /(Parameter left, Parameter right)
        {
            CheckForIncompatibleOperands(left, right, "divide");

            var newWeights = left._weights.Zip(right._weights, Matrix.op_DotDivide);
            var newBiases = left._biases.Zip(right._biases, Vector.op_DotDivide);

            return new Parameter(newWeights, newBiases);
        }

        public Parameter Pow(double power)
        {
            var newWeights = _weights.Select(w => w.PointwisePower(power));
            var neweBiases = _biases.Select(b => b.PointwisePower(power));

            return new(newWeights, neweBiases);
        }

        public Parameter Add(double scalar)
        {
            var newWeights = _weights.Select(w => w.Add(scalar));
            var newBiases = _biases.Select(b => b.Add(scalar));

            return new Parameter(newWeights, newBiases);
        }

        public double SquaredNorm()
            => _weights.Select(w => Math.Pow(w.FrobeniusNorm(), 2)).Sum()
                + _biases.Select(b => b.DotProduct(b)).Sum();

        public void SetWeightsUnivariate(Activation[] activators, IEnumerable<Vector> inputs, double varianceTolerance, int maxIterations)
        {
            for (int layer = 0; layer < LayerCount; layer++)
            {
                for (int iter = 1; iter <= maxIterations; iter++)
                {
                    double variance = inputs
                        .Select(input => Layers(input, activators, out _)[layer].Variance())
                        .Average();

                    if (Math.Abs(variance - 1) >= varianceTolerance)
                        _weights[layer] /= Math.Sqrt(variance);
                    else break;
                }
            }
        }

        private Vector[] Layers(Vector input, Activation[] activators, out Vector[] layersBeforeActivation)
        {
            if (input.Count != _weights[0].ColumnCount)
                throw new ArithmeticException($"The input has dimension {input.Count}, which does not match the required dimension of {_weights[0].ColumnCount}");

            if (LayerCount != activators.Length)
                throw new ArgumentException($"{activators.Length} activators were supplied, which does not match the amount {LayerCount} of layers of the parameter object.");

            Vector[] layers = new Vector[LayerCount];
            layersBeforeActivation = new Vector[LayerCount];
            Vector prevLayer = input;

            for (int i = 0; i < LayerCount; i++)
            {
                Vector beforeActivation = _weights[i] * prevLayer + _biases[i];
                layersBeforeActivation[i] = beforeActivation;
                layers[i] = activators[i].Apply(beforeActivation);

                prevLayer = layers[i];
            }
            return layers;
        }


        private static Matrix[] ActivatorDerivs(Vector[] layersBeforeActivation, Activation[] activators)
            => activators
                .Zip(layersBeforeActivation, (activator, layer) => activator.ApplyDerivative(layer))
                .ToArray();


        public Vector Output(Vector input, Activation[] activators)
            => Layers(input, activators, out _).Last();


        private static Matrix CostGradWrtWeight(Vector costGradWrtLayer, Matrix activatorDerivs, Vector layerBehind)
        {
            Vector derivs = activatorDerivs.TransposeThisAndMultiply(costGradWrtLayer);
            return Vector.OuterProduct(derivs, layerBehind);
        }

        private static Vector CostGradWrtBias(Vector costGradWrtLayer, Matrix activatorDerivs)
            => activatorDerivs.TransposeThisAndMultiply(costGradWrtLayer);

        private static Vector CostGradWrtLayerBehind(Vector costGradWrtLayer, Matrix activatorDerivs, Matrix layerWeight)
        {
            Matrix derivsFromLayer = activatorDerivs * layerWeight;
            return derivsFromLayer.TransposeThisAndMultiply(costGradWrtLayer);
        }

        
        public Parameter CostGrad(Vector input, Vector desiredOutput, Activation[] activators, CostFunction cost)
        {
            Vector[] layers = Layers(input, activators, out Vector[] layersBeforeActivation);
            //if (!layers.All(VectorFunctions.IsFinite))
            //    throw new ArithmeticException($"layers are non-finite");

            Matrix[] activatorDerivs = ActivatorDerivs(layersBeforeActivation, activators);
            //if (!activatorDerivs.All(MatrixFunctions.IsFinite))
              //  throw new ArithmeticException($"activator derivs are non-finite");

            Vector costGradWrtLayer = cost.Derivative(layers.Last(), desiredOutput);
            //if (!VectorFunctions.IsFinite(costGradWrtLayer))
              //  throw new ArithmeticException($"Cost grad wrt layer is non-finite");
            
            Matrix[] weightCostGrads = new Matrix[LayerCount];
            Vector[] biasCostGrads = new Vector[LayerCount];

            for (int i = LayerCount - 1; i >= 0; i--)
            {
                Vector layerBehind = i > 0 ? layers[i - 1] : input;
                // weightCostGrads[i] = CostGradWrtWeight(costGradWrtLayer, activatorDerivs[i], layerBehind);
                biasCostGrads[i] = CostGradWrtBias(costGradWrtLayer, activatorDerivs[i]);
                //if (!VectorFunctions.IsFinite(biasCostGrads[i]))
                  //  throw new ArithmeticException($"bias cost grad {i} is non-finite");

                weightCostGrads[i] = Vector.OuterProduct(biasCostGrads[i], layerBehind);
                //if (!MatrixFunctions.IsFinite(weightCostGrads[i]))
                  //  throw new ArithmeticException($"weight cost grad {i} is non-finite");

                costGradWrtLayer = CostGradWrtLayerBehind(costGradWrtLayer, activatorDerivs[i], _weights[i]);
                //if (!VectorFunctions.IsFinite(costGradWrtLayer))
                  //  throw new ArithmeticException($"cost grad wrt layer in step {i} is non-finite");
            }
            Parameter grad = new(weightCostGrads, biasCostGrads);
            if (!grad.IsFinite())
                throw new ArithmeticException($"Gradient as calculated in Parameter.cs is non-finite");

            return grad;
        }
        
        public Parameter DeepCopy()
        {
            var newWeights = _weights.Select(w => w.Clone());
            var newBiases = _biases.Select(b => b.Clone());
            return new Parameter(newWeights, newBiases);
        }

        public bool IsFinite()
            => _biases.All(VectorFunctions.IsFinite) && _weights.All(MatrixFunctions.IsFinite);
    }
}
