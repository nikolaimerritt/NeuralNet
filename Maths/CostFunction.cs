using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetLearning.Maths;

namespace NeuralNetLearning
{
    using Vector = Vector<double>;
    public abstract class CostFunction
    {
        public abstract double Apply(Vector estimated, Vector actual);
        
        public abstract Vector Derivative(Vector estimated, Vector actual);
        public virtual void WriteToFile(string filepath)
            => HyperParamEncoder.EncodeToFile(this.GetType().Name, filepath);

        public static CostFunction ReadFromFile(string filepath)
        {
            if (!File.Exists(filepath))
                throw new FileNotFoundException($"Could not find file {filepath}");

            string typeName = HyperParamEncoder.DecodeTypeName(filepath);

            if (typeName == nameof(MSECost))
                return new MSECost();

            if (typeName == nameof(CrossEntropyCost))
                return new CrossEntropyCost();

            if (typeName == nameof(HuberCost))
                return new HuberCost(HyperParamEncoder.Decode(filepath, "outlier boundary"));

            else throw new ArgumentException($"Could not recognise a cost function from the name {typeName}");
        }
    }

    public class MSECost : CostFunction
    {
        public MSECost() { }

        public override double Apply(Vector estimated, Vector actual)
        {
            Vector error = actual - estimated;
            return error.DotProduct(error) / error.Count;
        }

        public override Vector Derivative(Vector estimated, Vector actual)
        {
            Vector error = actual - estimated;
            return 2 * error * (-1) / error.Count;
        }
    }

    public class CrossEntropyCost : CostFunction
    {
        private const double avoidDivisionByZero = 1e-8;
        public CrossEntropyCost() { }

        public override double Apply(Vector estimated, Vector actual)
            => -actual.DotProduct(estimated.Add(avoidDivisionByZero).PointwiseLog());

        public override Vector Derivative(Vector estimated, Vector actual)
            => -Vector.op_DotDivide(actual, estimated.Add(avoidDivisionByZero));
    }

    public class HuberCost : CostFunction
    {
        private readonly double _outlierBoundary;
        public HuberCost(double outlierBoundary = 1)
            => _outlierBoundary = outlierBoundary;

        public override double Apply(Vector<double> estimated, Vector<double> actual)
        {
            double cost = 0;
            for (int i = 0; i < estimated.Count; i++)
            {
                double error = actual[i] - estimated[i];
                if (-_outlierBoundary < error && error < _outlierBoundary)
                    cost += 0.5 * Math.Pow(error, 2) / estimated.Count;
                else
                    cost += _outlierBoundary * Math.Abs(error) / estimated.Count - 0.5 * Math.Pow(_outlierBoundary, 2) / estimated.Count;
            }
            return cost;
        }

        public override Vector Derivative(Vector estimated, Vector actual)
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

        public override void WriteToFile(string filepath)
            => HyperParamEncoder.EncodeToFile(this.GetType().Name, filepath, ("outlier boundary", _outlierBoundary));
    }
}
