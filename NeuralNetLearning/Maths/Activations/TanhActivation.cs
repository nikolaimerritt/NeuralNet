using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace NeuralNetLearning.Maths.Activations
{
    using Matrix = Matrix<double>;
    using Vector = Vector<double>;
    public class TanhActivation : Activation
    {
        private Vector _cachedInput;
        private Vector _cachedTanh;
        public TanhActivation() { }

        public override Vector Apply(Vector input)
        {
            double[] activation = new double[input.Count];
            for (int i = 0; i < input.Count; i++)
                activation[i] = Math.Tanh(input[i]);

            _cachedInput = input;
            _cachedTanh = new DenseVector(activation);
            return _cachedTanh;
        }

        public override Matrix ApplyDerivative(Vector input)
        {
            double[] derivEntries = new double[input.Count];
            Vector tanh = input == _cachedInput ? _cachedTanh : input.PointwiseTanh();

            for (int i = 0; i < derivEntries.Length; i++)
                derivEntries[i] = 1 - Math.Pow(tanh[i], 2);

            return new DiagonalMatrix(derivEntries.Length, derivEntries.Length, derivEntries);
        }
    }
}
