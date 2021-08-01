using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;

namespace Maths
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;

    static class ScalarFunctions
    {
        public static readonly double Epsilon = 1e-7;
        public static double Relu(double x)
            => x >= 0 ? x : 0;

        public static double ReluDeriv(double x)
        {
            if (x < 0)
                return 0;
            if (x == 0)
                return 0.5;
            else
                return 1;
        }

        public static double Sigmoid(double x)
            => 1 / (1 + Math.Exp(-x));

        public static double SigmoidDeriv(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid * (1 - sigmoid);
        }

        public static double NumericDerivative(Func<double, double> f, double differentiateAt)
        {
            double difference = f(differentiateAt + Epsilon) - f(differentiateAt - Epsilon);
            return difference / (2 * Epsilon);
        }

        public static double NumericDerivative(Func<Vector, double> f, Vector differentiateAt, Vector direction)
        {
            double difference = f(differentiateAt + Epsilon * direction) - f(differentiateAt - Epsilon * direction);
            return difference / (2 * Epsilon);
        }

        public static double NumericDerivative(Func<Matrix, double> f, Matrix differentiateAt, Matrix direction)
        {
            double difference = f(differentiateAt + Epsilon * direction) - f(differentiateAt - Epsilon * direction);
            return difference / (2 * Epsilon);
        }
    }

}
