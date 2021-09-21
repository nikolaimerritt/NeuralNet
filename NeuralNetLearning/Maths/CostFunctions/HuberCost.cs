using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetLearning.Serialization;
using System;

namespace NeuralNetLearning.Maths.CostFunctions
{
    public class HuberCost : CostFunction
    {
        [SerializableHyperParameter("outlier boundary")]
        private readonly double _outlierBoundary = 1;
        public HuberCost(double outlierBoundary)
            => _outlierBoundary = outlierBoundary;

        public HuberCost()
        { }

        public override double Apply(Vector<double> estimated, Vector<double> actual)
        {
            double cost = 0;
            for (int i = 0; i < estimated.Count; i++)
            {
                double error = actual[i] - estimated[i];
                if (-_outlierBoundary < error && error < _outlierBoundary)
                    cost += 0.5 * Math.Pow(error, 2) / estimated.Count;
                else
                    cost += _outlierBoundary * (Math.Abs(error) - 0.5 * _outlierBoundary) / estimated.Count;
            }
            return cost;
        }

        public override Vector<double> Derivative(Vector<double> estimated, Vector<double> actual)
        {
            double[] derivs = new double[estimated.Count];
            for (int i = 0; i < derivs.Length; i++)
            {
                double error = actual[i] - estimated[i];
                if (-_outlierBoundary < error && error < _outlierBoundary)
                    derivs[i] = -error / estimated.Count;
                else
                    derivs[i] = -_outlierBoundary * Math.Sign(error) / estimated.Count;
            }

            return new DenseVector(derivs);
        }

        //public override void WriteToFile(string filepath)
          //  => HyperParamEncoder.EncodeToFile(this.GetType().Name, filepath, ("outlier boundary", _outlierBoundary));
    }
}
