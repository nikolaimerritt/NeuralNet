using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLearning
{
	using VectorFn = Func<Vector<double>, Vector<double>>;

	public class NeuralLayerConfig
	{
		public readonly VectorFn Activator;
		public readonly VectorFn ActivatorDeriv;

		public static VectorFn PiecewiseVectorFn(Func<double, double> applyToEachEl)
        {
			return new VectorFn(vector =>
				Vector<double>.Build.DenseOfEnumerable(vector.Select(applyToEachEl))
			);
        }

		public NeuralLayerConfig(VectorFn activator, VectorFn activatorDeriv)
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
			activatorDeriv: PiecewiseVectorFn(x => Sigmoid(x) * (1 - Sigmoid(x)))
		);

		public static readonly NeuralLayerConfig IdentityConfig = new
		(
			activator: new VectorFn(vector => vector),
			activatorDeriv: PiecewiseVectorFn(x => 1)
		);
    }
}
