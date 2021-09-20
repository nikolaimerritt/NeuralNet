using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Linq;

namespace NeuralNetLearning.Maths.Activations
{
    public class SoftmaxActivation : Activation
    {
        private Vector<double> _cachedInput;
        private Vector<double> _cachedSoftmax;

        public SoftmaxActivation() { }

        public override Vector<double> Apply(Vector<double> input)
        {
            double maxInput = input.Max();
            double[] exponentials = new double[input.Count];
            for (int i = 0; i < exponentials.Length; i++)
                exponentials[i] = Math.Exp(input[i] - maxInput);

            _cachedInput = input;
            _cachedSoftmax = new DenseVector(exponentials) / exponentials.Sum();
            return _cachedSoftmax;
        }

        public override Matrix<double> ApplyDerivative(Vector<double> input)
        {
            Vector<double> softmax = input == _cachedInput ? _cachedSoftmax : Apply(input);
            DiagonalMatrix softmaxAlongDiagonal = new(softmax.Count, softmax.Count, softmax.ToArray());
            return -Vector<double>.OuterProduct(softmax, softmax) + softmaxAlongDiagonal;
        }
    }
}
