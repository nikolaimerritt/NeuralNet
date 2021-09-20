using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLearning.Maths.Activations
{
    public class IdentityActivation : Activation
    {
        public IdentityActivation() { }

        public override Vector<double> Apply(Vector<double> input)
            => input;

        public override Matrix<double> ApplyDerivative(Vector<double> input)
            => Matrix<double>.Build.DenseIdentity(input.Count);
    }
}
