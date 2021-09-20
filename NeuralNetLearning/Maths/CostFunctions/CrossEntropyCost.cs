using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using System.Threading.Tasks;

namespace NeuralNetLearning.Maths.CostFunctions
{
    public class CrossEntropyCost : CostFunction
    {
        private const double avoidDivisionByZero = 1e-8;
        public CrossEntropyCost() { }

        public override double Apply(Vector<double> estimated, Vector<double> actual)
            => -actual.DotProduct(estimated.Add(avoidDivisionByZero).PointwiseLog());

        public override Vector<double> Derivative(Vector<double> estimated, Vector<double> actual)
            => -Vector<double>.op_DotDivide(actual, estimated.Add(avoidDivisionByZero));
    }
}
