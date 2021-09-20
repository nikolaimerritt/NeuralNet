using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetLearning.Serialization;

namespace NeuralNetLearning.Maths.Activations
{
    public class ReluActivation : Activation
    {
        private readonly double _leak;
        public ReluActivation(double leak = 0)
        {
            _leak = leak;
        }

        public override Vector<double> Apply(Vector<double> input)
        {
            double[] activation = new double[input.Count];
            for (int i = 0; i < activation.Length; i++)
                activation[i] = input[i] > 0 ? input[i] : _leak * input[i];

            return new DenseVector(activation);
        }

        public override Matrix<double> ApplyDerivative(Vector<double> input)
        {
            double[] deriv = new double[input.Count];
            for (int i = 0; i < deriv.Length; i++)
                if (input[i] > 0)
                    deriv[i] = 1;
                else if (input[i] == 0)
                    deriv[i] = (1 + _leak) / 2;
                else
                    deriv[i] = _leak;

            return new DiagonalMatrix(deriv.Length, deriv.Length, deriv);
        }

        public override void WriteToFile(string filepath)
            => HyperParamEncoder.EncodeToFile(this.GetType().Name, filepath, ("leak", _leak));
    }
}
