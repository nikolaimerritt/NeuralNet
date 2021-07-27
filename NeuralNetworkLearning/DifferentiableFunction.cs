using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Maths;

namespace NeuralNetLearning
{
	using Vector = Vector<double>;

	public class DifferentiableFunction
	{
		public readonly Func<Vector, Vector> Apply;
		public readonly Func<Vector, Vector> ApplyDerivative;

		public DifferentiableFunction(Func<Vector, Vector> activator, Func<Vector, Vector> activatorDeriv)
		{
			Apply = activator;
			ApplyDerivative = activatorDeriv;
		}


		public static readonly DifferentiableFunction Relu = new
		(
			activator: VectorFunctions.Piecewise(ScalarFunctions.Relu), 
			activatorDeriv: VectorFunctions.Piecewise(ScalarFunctions.ReluDeriv)
		);


		public static readonly DifferentiableFunction Sigmoid = new
		(
			activator: VectorFunctions.Piecewise(ScalarFunctions.Sigmoid),
			activatorDeriv: VectorFunctions.Piecewise(ScalarFunctions.SigmoidDeriv)	
		);


		public static readonly DifferentiableFunction Identity = new
		(
			activator: new Func<Vector, Vector>(vector => vector),
			activatorDeriv: VectorFunctions.Piecewise(x => 1)
		);
    }
}
