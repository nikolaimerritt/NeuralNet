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
        public static double MSE(Vector obtainedValue, Vector desiredValue)
            => (obtainedValue - desiredValue).DotProduct(obtainedValue - desiredValue) / obtainedValue.Count;


        public static Vector MSEderiv(Vector obtainedValue, Vector desiredValue)
            => 2 * (obtainedValue - desiredValue);


        public static Func<Vector, Vector> Piecewise(Func<double, double> scalarFn)
        {
            return new (vec => Vector<double>.Build.DenseOfEnumerable(vec.Select(scalarFn)));
        }


        public static Vector StdNormal(int dim) // Normal(mean = 0, var = 1)
            => MatrixFunctions.StdNormal(1, dim).Column(0);


        public static Vector StdUniform(int dim) // [-1, 1]
            => MatrixFunctions.StdUniform(1, dim).Column(0);


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
    }
}
