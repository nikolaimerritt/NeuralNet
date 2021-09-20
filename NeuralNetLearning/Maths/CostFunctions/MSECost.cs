using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLearning.Maths.CostFunctions
{
    public class MSECost : CostFunction
    {
        public MSECost() { }

        public override double Apply(Vector<double> estimated, Vector<double> actual)
        {
            Vector<double> error = actual - estimated;
            return error.DotProduct(error) / error.Count;
        }

        public override Vector<double> Derivative(Vector<double> estimated, Vector<double> actual)
        {
            Vector<double> error = actual - estimated;
            return 2 * error * (-1) / error.Count;
        }
    }
}
