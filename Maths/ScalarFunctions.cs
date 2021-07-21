using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace Maths
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;

    static class ScalarFunctions
    {
        public static double Relu(double x)
            => x >= 0 ? x : 0;

        public static double ReluDeriv(double x)
            => x >= 0 ? 1 : 0;

        public static double Sigmoid(double x)
            => 1 / (1 + Math.Exp(-x));

        public static double SigmoidDeriv(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid * (1 - sigmoid);
        }
    }

}
