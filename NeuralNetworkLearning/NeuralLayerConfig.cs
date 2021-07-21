using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Maths;

namespace NeuralNetLearning
{
	using Vector = Vector<double>;

	public class NeuralLayerConfig
	{
		public readonly Func<Vector, Vector> Activator;
		public readonly Func<Vector, Vector> ActivatorDeriv;

		public NeuralLayerConfig(Func<Vector, Vector> activator, Func<Vector, Vector> activatorDeriv)
		{
			Activator = activator;
			ActivatorDeriv = activatorDeriv;
		}


		public static readonly NeuralLayerConfig ReluConfig = new
		(
			activator: VectorFunctions.Piecewise(ScalarFunctions.Relu), 
			activatorDeriv: VectorFunctions.Piecewise(ScalarFunctions.ReluDeriv)
		);


		public static readonly NeuralLayerConfig SigmoidConfig = new
		(
			activator: VectorFunctions.Piecewise(ScalarFunctions.Sigmoid),
			activatorDeriv: VectorFunctions.Piecewise(ScalarFunctions.SigmoidDeriv)	
		);


		public static readonly NeuralLayerConfig IdentityConfig = new
		(
			activator: new(vector => vector),
			activatorDeriv: VectorFunctions.Piecewise(x => 1)
		);


    }
}
