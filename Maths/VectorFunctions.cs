using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace Maths
{
    using static MatrixFunctions;
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;

    static class VectorFunctions
    {
        private static readonly VectorBuilder<double> builder = Vector.Build;
        public static double MSE(Vector obtainedValue, Vector desiredValue)
            => (obtainedValue - desiredValue).DotProduct(obtainedValue - desiredValue) / obtainedValue.Count;


        public static Vector MSEderiv(Vector obtainedValue, Vector desiredValue)
            => 2 * (obtainedValue - desiredValue) / obtainedValue.Count;


        public static Func<Vector, Vector> Piecewise(Func<double, double> scalarFn)
        {
            return new (vec => builder.DenseOfEnumerable(vec.Select(scalarFn)));
        }


        public static Vector StdNormal(int dim) // Normal(mean = 0, var = 1)
            => MatrixFunctions.StdNormal(dim, 1).Column(0);


        public static Vector StdUniform(int dim) // [-1, 1]
            => MatrixFunctions.StdUniform(dim, 1).Column(0);


        public static Vector Read(string filepath)
        {
            Matrix vectorAsMatrix = MatrixFunctions.Read(filepath);
            if (vectorAsMatrix.RowCount == 1)
            {
                return vectorAsMatrix.Row(0);
            }
            else if (vectorAsMatrix.ColumnCount == 1)
            {
                return vectorAsMatrix.Column(0);
            }
            else
            {
                throw new Exception($"Could not read vector from file {filepath} as there are multiple rows and multiple columns in the file.");
            }
        }

        public static void Write(this Vector vector, string filepath)
        {
            Matrix vectorAsColMatrix = CreateMatrix.DenseOfColumnVectors(vector);
            vectorAsColMatrix.Write(filepath);
        }

        public static Vector BasisVector(int length, int oneIdx)
            => MatrixFunctions.BasisMatrix(rows: length, cols: 1, oneRow: oneIdx, oneCol: 0).Column(0);

        public static Vector[] BasisVectors(int length)
        {
            List<Vector> basisVectors = new (length);
            for (int oneIdx = 0; oneIdx < length; oneIdx++)
            {
                basisVectors.Add(BasisVector(length, oneIdx));
            }
            return basisVectors.ToArray();
        }

        public static double epsilon = ScalarFunctions.Epsilon;


        public static Vector NumericPartialDerivs(Func<Vector, double> f, Vector differentiateAt)
        {
            List<double> partialDerivs = new();
            foreach (Vector direction in BasisVectors(differentiateAt.Count))
            {
                double difference = f(differentiateAt + epsilon * direction) - f(differentiateAt - epsilon * direction);
                partialDerivs.Add(difference / (2 * epsilon));
            }
            return builder.DenseOfEnumerable(partialDerivs);
        }
    }
}
