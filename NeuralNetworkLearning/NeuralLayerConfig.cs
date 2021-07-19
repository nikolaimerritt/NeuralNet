using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLearning
{
	using Vector = Vector<double>;

	public class NeuralLayerConfig
	{
		public readonly Func<Vector, Vector> Activator;
		public readonly Func<Vector, Vector> ActivatorDeriv;

		public static Func<Vector, Vector> PiecewiseVectorFn(Func<double, double> applyToEachEl)
        {
			return new(vec => Vector<double>.Build.DenseOfEnumerable(vec.Select(applyToEachEl)));
        }

		public NeuralLayerConfig(Func<Vector, Vector> activator, Func<Vector, Vector> activatorDeriv)
		{
			Activator = activator;
			ActivatorDeriv = activatorDeriv;
		}

		private static double Heaviside(double x) => x > 0 ? 1.0 : 0.0;

		private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));


		public static readonly NeuralLayerConfig ReluConfig = new
		(
			activator: PiecewiseVectorFn(x => x * Heaviside(x)), 
			activatorDeriv: PiecewiseVectorFn(Heaviside)
		);

		public static readonly NeuralLayerConfig SigmoidConfig = new
		(
			activator: PiecewiseVectorFn(Sigmoid),
			activatorDeriv: PiecewiseVectorFn(x => Sigmoid(x) * (1 - Sigmoid(x)))	// will prob cache Sigmoid(x) so this isnt silly
		);

		public static readonly NeuralLayerConfig IdentityConfig = new
		(
			activator: new (vector => vector),
			activatorDeriv: PiecewiseVectorFn(x => 1)
		);

		public static double MSE(Vector obtainedValue, Vector desiredValue)
			=> (obtainedValue - desiredValue).DotProduct(obtainedValue - desiredValue);

		public static Vector MSEderiv(Vector obtainedValue, Vector desiredValue)
			=> 2 * (obtainedValue - desiredValue);
    }
}
