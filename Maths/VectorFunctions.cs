using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;

namespace Maths
{
    using static MatrixFunctions;
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;

    public static class VectorFunctions
    {
        private static readonly VectorBuilder<double> builder = Vector.Build;
        public static double MSE(Vector obtainedValue, Vector desiredValue)
            => (obtainedValue - desiredValue).DotProduct(obtainedValue - desiredValue) / obtainedValue.Count;

        public static Vector MSEderiv(Vector obtainedValue, Vector desiredValue)
            => 2 * (obtainedValue - desiredValue) / obtainedValue.Count;

        private static Vector Relu(this Vector input)
        {
            double[] relu = new double[input.Count];
            for (int i = 0; i < relu.Length; i++)
                relu[i] = input[i] > 0 ? input[i] : 0;

            return new DenseVector(relu);
        }

        private static Vector ReluDeriv(this Vector input)
        {
            double[] reluDeriv = new double[input.Count];
            for (int i = 0; i < reluDeriv.Length; i++)
                reluDeriv[i] = input[i] > 0 ? 1.0 : 0;

            return new DenseVector(reluDeriv);
        }

        private static Vector TanhSigmoid(this Vector input)
        {
            double[] sigmoid = new double[input.Count];
            for (int i = 0; i < input.Count; i++)
                sigmoid[i] = Math.Tanh(input[i]);

            return new DenseVector(sigmoid);
        }

        private static Vector TanhSigmoidDeriv(this Vector input)
        {
            double[] sigmoidDeriv = new double[input.Count];
            for (int i = 0; i < input.Count; i++)
                sigmoidDeriv[i] = 1 - Math.Pow(Math.Tanh(input[i]), 2);

            return new DenseVector(sigmoidDeriv);
        }

        public static Vector IdentitDeriv(this Vector input)
            => DenseVector.Create(input.Count, 1.0);

        public static Vector Activate(this Vector input, Activation activation)
            => activation switch
            {
                Activation.Identity => input,
                Activation.Relu => input.Relu(),
                Activation.TanhSigmoid => input.TanhSigmoid(),
                _ => throw new ArgumentException($"Activation {activation} not recognised")
            };

        public static Vector ActivateDerivative(this Vector input, Activation activation)
            => activation switch
            {
                Activation.Identity => input.IdentitDeriv(),
                Activation.Relu => input.ReluDeriv(),
                Activation.TanhSigmoid => input.TanhSigmoidDeriv(),
                _ => throw new ArgumentException($"Activation {activation} not recognised")
            };


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
    }
}
