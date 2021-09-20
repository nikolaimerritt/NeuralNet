using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using NeuralNetLearning.Maths.Activations;
using NeuralNetLearning.Maths.CostFunctions;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetLearning.Maths
{
    using Matrix = Matrix<double>;
    using Vector = Vector<double>;
    internal class Parameter
    {
        /// <summary>
        /// The number of active (i.e. non-input) layers being simulated. This is equal to the number of weight matrices, which is turn is equal to the number of bias vectors.
        /// </summary>
        public int ActiveLayerCount
        {
            get => _weights.Length;
        }

        /// <summary>
        /// The size of each layer, in order of calculation. This is the size of the input, hidden, and output layers.
        /// </summary>
        public int[] LayerSizes
        {
            get 
            {
                List<int> layerSizes = new() { _weights.First().ColumnCount };
                layerSizes.AddRange(_weights.Select(w => w.RowCount));
                return layerSizes.ToArray();
            }
        }

        /// <summary>
        /// The total number of scalar entries in the weight matrices and bias vectors.
        /// </summary>
        public int EntriesCount
        {
            get => 
                _weights.Select(w => w.RowCount * w.ColumnCount).Sum() +
                _biases.Select(b => b.Count).Sum();
        }

        private readonly Matrix[] _weights;
        private readonly Vector[] _biases;

        /// <summary>
        /// Creates a new <see cref="Parameter"/> object that stores the supplied weight matrices and bias vectors.
        /// </summary>
        /// <param name="weights">The weight matrices the new Paramter object will store. A shallow copy of the <c>IEnumerable</c> is created.</param>
        /// <param name="biases">The bias vectors the new Parameter object will store. A shallow copy of the <c>IEnumerable</c> is created.</param>
        public Parameter(IEnumerable<Matrix> weights, IEnumerable<Vector> biases)
        {
            CheckForConstructionError(weights, biases);
            _weights = weights.ToArray();
            _biases = biases.ToArray();
        }

        /// <summary>
        /// Writes the weight matrices and bias vectors of the current <see cref="Parameter"/> to individual plain text files in <paramref name="directoryPath"/>. The weights and biases are written in a human-readable format.
        /// </summary>
        /// <param name="directoryPath"></param>
        public void WriteToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            for (int i = 0; i < ActiveLayerCount; i++)
            {
                string weightPath = $"{directoryPath}/weight {i + 1}.csv";
                _weights[i].Write(weightPath);
                string biasPath = $"{directoryPath}/bias {i + 1}.csv";
                _biases[i].Write(biasPath);
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

        /// <summary>
        /// Adds the weights and biases of <paramref name="other"/> directly to the weights and biases of the current <see cref="Parameter"/>. 
        /// Updates the current<see cref="Parameter"/>'s weights and biases. Is more memory efficient than invoking <c>+=</c>.
        /// </summary>
        /// <param name="other"> The <see cref="Parameter"/> to be added component-wise. Is unaffected. </param>
        /// <exception cref="ArithmeticException">Raises if <paramref name="other"/> has different layer sizes to the current <see cref="Parameter"/>.</exception>
        public void InPlaceAdd(Parameter other)
        {
            if (!LayerSizes.SequenceEqual(other.LayerSizes))
                throw new ArithmeticException($"Could not add two parameters with different layer sizes.");

            for (int l = 0; l < ActiveLayerCount; l++)
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

        /// <summary>
        /// Adds <paramref name="scalar"/> to every weight and bias entry of the current <see cref="Parameter"/>. Updates the current <see cref="Parameter"/>'s weights and biases. Is more memory efficient than using <c>+=</c>.
        /// </summary>
        /// <param name="scalar">Is added to every weight and bias entry.</param>
        public void InPlaceAdd(double scalar)
        {
            for (int l = 0; l < ActiveLayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] += scalar;
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] += scalar;
                }
            }
        }

        /// <summary>
        /// Subtracts the weights and biases of <paramref name="other"/> directly from the weights and biases of the current <see cref="Parameter"/>. Updates the current <see cref="Parameter"/>'s weights and biases. Is more memory efficient than using `-=`.
        /// </summary>
        /// <param name="other"> The Parameter to be added component-wise. Is unaffected by the addition. </param>
        /// <exception cref="ArithmeticException">Raises if <paramref name="other"/> has different layer sizes to the current <see cref="Parameter"/>.</exception>
        public void InPlaceSubtract(Parameter other)
        {
            if (!LayerSizes.SequenceEqual(other.LayerSizes))
                throw new ArithmeticException($"Could not add two parameters with different layer sizes.");

            for (int l = 0; l < ActiveLayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] -= other._weights[l][r, c];
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] -= other._biases[l][i];
                }
            }
        }

        /// <summary>
        /// Subtracts <paramref name="scalar"/> from every weight and bias entry of the current <see cref="Parameter"/>. 
        /// Updates the current <see cref="Parameter"/>'s weights and biases. Is more memory efficient than using <c>-=</c>.
        /// </summary>
        /// <param name="scalar">Is subtracted from every weight and bias entry.</param>
        public void InPlaceSubtract(double scalar)
        {
            for (int l = 0; l < ActiveLayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] -= scalar;
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] -= scalar;
                }
            }
        }

        /// <summary>
        /// Multiplies the weights and biases in the current <see cref="Parameter"/> by the corresponding weights and biases in <paramref name="other"/>. Updates the current <see cref="Parameter"/>'s weights and biases. Is more memory efficient than using `*=`.
        /// </summary>
        /// <param name="other"> The Parameter that multiplies the current <see cref="Parameter"/> component-wise. Is unaffected by the component-wise multiplication. </param>
        /// <exception cref="ArithmeticException">Raises if <paramref name="other"/> has different layer sizes to the current <see cref="Parameter"/>.</exception>
        public void InPlaceMultiply(Parameter other)
        {
            if (!LayerSizes.SequenceEqual(other.LayerSizes))
                throw new ArithmeticException($"Could not multiply two parameters with different layer sizes.");

            for (int l = 0; l < ActiveLayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] *= other._weights[l][r, c];
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] *= other._biases[l][i];
                }
            }
        }

        /// <summary>
        /// Multiplies every weight and bias entry of the current <see cref="Parameter"/> by <paramref name="scalar"/>. Updates the current <see cref="Parameter"/>'s weights and biases. Is more memory efficient than using <c>*=</c>.
        /// </summary>
        /// <param name="scalar">Multiplies every weight and bias entry.</param>
        public void InPlaceMultiply(double scalar)
        {
            for (int l = 0; l < ActiveLayerCount; l++)
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

        /// <summary>
        /// Divides the weights and biases in the current <see cref="Parameter"/> by the corresponding weights and biases in <paramref name="other"/>. Updates the current <see cref="Parameter"/>'s weights and biases. Is more memory efficient than using `/=`.
        /// </summary>
        /// <param name="other"> The <see cref="Parameter"/> that divides the current <see cref="Parameter"/> component-wise. Is unaffected by the component-wise division. </param>
        /// <exception cref="ArithmeticException">Raises if <paramref name="other"/> has different layer sizes to the current <see cref="Parameter"/>.</exception>
        public void InPlaceDivide(Parameter other)
        {
            if (!LayerSizes.SequenceEqual(other.LayerSizes))
                throw new ArithmeticException($"Could not divide two parameters with different layer sizes.");

            for (int l = 0; l < ActiveLayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] /= other._weights[l][r, c];
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] /= other._biases[l][i];
                }
            }
        }

        /// <summary>
        /// Divides every weight and bias entry of the current <see cref="Parameter"/> by <paramref name="scalar"/>. Updates the current <see cref="Parameter"/>'s weights and biases. Is more memory efficient than using <c>/=</c>.
        /// </summary>
        /// <param name="scalar">Divides every weight and bias entry.</param>
        /// <exception cref="DivideByZeroException">Raised if <paramref name="scalar"/> is zero.</exception>
        public void InPlaceDivide(double scalar)
        {
            if (scalar == 0)
                throw new DivideByZeroException($"Could not divide a parameter object by zero.");

            for (int l = 0; l < ActiveLayerCount; l++)
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

        /// <summary>
        /// Raises each weight and bias entry in the current <see cref="Parameter"/> by the power of the corresponding weight / bias entry in in <paramref name="other"/>. 
        /// That is, performs component-wise exponentiation, storing the result in the current <see cref="Parameter"/>. 
        /// </summary>
        /// <param name="other"> The <see cref="Parameter"/> that component-wise exponentiates the current <see cref="Parameter"/>. Is unaffected by the component-wise exponentiation. </param>
        /// <exception cref="ArithmeticException">Raises if <paramref name="other"/> has different layer sizes to the current <see cref="Parameter"/>.</exception>
        public void InPlacePower(Parameter power)
        {
            if (!LayerSizes.SequenceEqual(power.LayerSizes))
                throw new ArithmeticException($"Could not exponentiate two parameters with different layer sizes.");

            for (int l = 0; l < ActiveLayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] = Math.Pow(_weights[l][r, c], power._weights[l][r, c]);
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] = Math.Pow(_biases[l][i], power._biases[l][i]);
                }
            }
        }

        /// <summary>
        /// Raises every weight and bias entry of the current <see cref="Parameter"/> by the exponent <paramref name="power"/>. Updates the current <see cref="Parameter"/>'s weights and biases.
        /// </summary>
        /// <param name="power">Exponentiates every weight and bias entry.</param>
        public void InPlacePower(double power)
        {
            for (int l = 0; l < ActiveLayerCount; l++)
            {
                for (int r = 0; r < _weights[l].RowCount; r++)
                {
                    for (int c = 0; c < _weights[l].ColumnCount; c++)
                    {
                        _weights[l][r, c] = Math.Pow(_weights[l][r, c], power);
                    }
                }
                for (int i = 0; i < _biases[l].Count; i++)
                {
                    _biases[l][i] = Math.Pow(_biases[l][i], power);
                }
            }
        }

        public static Parameter operator +(Parameter left, Parameter right)
        {
            Parameter result = left.DeepCopy();
            result.InPlaceAdd(right);
            return result;
        }

        public static Parameter operator +(Parameter param, double scalar)
        {
            Parameter result = param.DeepCopy();
            result.InPlaceAdd(scalar);
            return result;
        }

        public static Parameter operator *(double scalar, Parameter parameter)
        {
            Parameter result = parameter.DeepCopy();
            result.InPlaceMultiply(scalar);
            return result;
        }

        public static Parameter operator *(Parameter left, Parameter right)
        {
            Parameter result = left.DeepCopy();
            result.InPlaceMultiply(right);
            return result;
        }

        public static Parameter operator -(Parameter parameter)
            => (-1.0) * parameter;

        public static Parameter operator -(Parameter left, Parameter right)
        {
            Parameter result = left.DeepCopy();
            result.InPlaceSubtract(right);
            return result;
        }

        public static Parameter operator /(Parameter parameter, double scalar)
        {
            Parameter result = parameter.DeepCopy();
            result.InPlaceDivide(scalar);
            return result;
        }


        public static Parameter operator /(Parameter left, Parameter right)
        {
            Parameter result = left.DeepCopy();
            result.InPlaceDivide(right);
            return result;
        }

        public Parameter Pow(double power)
        {
            Parameter result = this.DeepCopy();
            result.InPlacePower(power);
            return result;
        }

        /// <summary>
        /// Returns the sum of the squares of each scalar in the weights and biases.
        /// </summary>
        public double SquaredNorm()
            => _weights.Select(w => Math.Pow(w.FrobeniusNorm(), 2)).Sum()
                + _biases.Select(b => b.DotProduct(b)).Sum();

        /// <summary>
        /// Adjusts the weights until the average variance of the output vectors is sufficiently close to 1.
        /// <para>
        /// Follows the <see href="http://cmp.felk.cvut.cz/~mishkdmy/papers/mishkin-iclr2016.pdf">LSUV algorithm</see> using the current weight matrices instead of random-initialised ones.
        /// </para>
        /// </summary>
        /// <param name="activators"> The <see cref="Activation"/>s used in calculating layers. </param>
        /// <param name="inputs"> The input <see cref="Vector{Double}"/>s which the average variance of the output <see cref="Vector{Double}"/> is taken from.</param>
        /// <param name="varianceTolerance">The weights stop being adjusted once the average variance of the output vectors is between <c>1 - <paramref name="varianceTolerance"/></c> and <c>1 + <paramref name="varianceTolerance"/></c>. </param>
        /// <param name="maxIterations">The weights are adjusted at most <paramref name="maxIterations"/> times.</param>
        public void SetWeightsUnivariate(Activation[] activators, IEnumerable<Vector> inputs, double varianceTolerance, int maxIterations)
        {
            for (int layer = 0; layer < ActiveLayerCount; layer++)
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

            if (ActiveLayerCount != activators.Length)
                throw new ArgumentException($"{activators.Length} activators were supplied, which does not match the amount {ActiveLayerCount} of layers of the parameter object.");

            Vector[] layers = new Vector[ActiveLayerCount];
            layersBeforeActivation = new Vector[ActiveLayerCount];
            Vector prevLayer = input;

            for (int i = 0; i < ActiveLayerCount; i++)
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


        public Vector GetOutputVector(Vector input, Activation[] activators)
            => Layers(input, activators, out _).Last();

        private static Vector CostGradWrtBias(Vector costGradWrtLayer, Matrix activatorDerivs)
            => activatorDerivs.TransposeThisAndMultiply(costGradWrtLayer);

        private static Vector CostGradWrtLayerBehind(Vector costGradWrtLayer, Matrix activatorDerivs, Matrix layerWeight)
        {
            Matrix derivsFromLayer = activatorDerivs * layerWeight;
            return derivsFromLayer.TransposeThisAndMultiply(costGradWrtLayer);
        }

        /// <summary>
        /// Each entry in the returned <see cref="Parameter"/> is the derivative of the cost function with respect to the corresponding entry in the current <see cref="Parameter"/>.
        /// </summary>
        /// <param name="input">The input vector.</param>
        /// <param name="desiredOutput">The expected output vector. Is compared to the calculated output vector in <paramref name="cost"/>.</param>
        /// <param name="activators"> The activators used in calculating the layers. </param>
        /// <param name="cost">The cost function which compares the calculated output vector to <paramref name="desiredOutput"/>.</param>
        public Parameter CostGradient(Vector input, Vector desiredOutput, Activation[] activators, CostFunction cost)
        {
            Vector[] layers = Layers(input, activators, out Vector[] layersBeforeActivation);
            Matrix[] activatorDerivs = ActivatorDerivs(layersBeforeActivation, activators);
            Vector costGradWrtLayer = cost.Derivative(layers.Last(), desiredOutput);
            
            Matrix[] weightCostGrads = new Matrix[ActiveLayerCount];
            Vector[] biasCostGrads = new Vector[ActiveLayerCount];

            for (int i = ActiveLayerCount - 1; i >= 0; i--)
            {
                Vector layerBehind = i > 0 ? layers[i - 1] : input;
                biasCostGrads[i] = CostGradWrtBias(costGradWrtLayer, activatorDerivs[i]);
                weightCostGrads[i] = Vector.OuterProduct(biasCostGrads[i], layerBehind);
                costGradWrtLayer = CostGradWrtLayerBehind(costGradWrtLayer, activatorDerivs[i], _weights[i]);
            }
            return new Parameter(weightCostGrads, biasCostGrads);
        }
        
        /// <summary>
        /// Returns a new <see cref="Parameter"/> object with weights and biases that are deep copies of the weights and biases of the current <see cref="Parameter"/>.
        /// Changes to the current <see cref="Parameter"/> will not affect the new <see cref="Parameter"/>.
        /// </summary>
        public Parameter DeepCopy()
        {
            var newWeights = _weights.Select(w => w.Clone());
            var newBiases = _biases.Select(b => b.Clone());
            return new Parameter(newWeights, newBiases);
        }

        /// <summary>
        /// Checks if each entry is finite: i.e. non-infinite and non-NaN.
        /// </summary>
        public bool IsFinite()
            => _biases.All(VectorFunctions.IsFinite) && _weights.All(MatrixFunctions.IsFinite);
    }
}
