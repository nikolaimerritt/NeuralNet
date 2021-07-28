using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.Distributions;

namespace Maths
{
	using Matrix = Matrix<double>;
    static class MatrixFunctions
    {
		private static readonly Normal _stdNormal = new (mean: 0, stddev: 1);
		private static readonly ContinuousUniform _stdUniform = new (lower: -1, upper: 1);
		private static readonly MatrixBuilder<double> _builder = Matrix.Build;


		public static Matrix StdNormal(int rows, int cols)
			=> Matrix.Build.Random(rows, cols, _stdNormal);


		public static Matrix StdUniform(int rows, int cols) // [-1, 1]
			=> Matrix.Build.Random(rows, cols, _stdUniform);


		public static Matrix Read(string filepath)
			=> DelimitedReader.Read<double>(filepath, delimiter: "\t");


		public static void Write(this Matrix matrix, string filepath)
			=> DelimitedWriter.Write(filepath, matrix, delimiter: "\t");


		public static Matrix BasisMatrix(int rows, int cols, int oneRow, int oneCol)
        {
			Matrix zeroes = _builder.Dense(rows, cols);
			zeroes[oneRow, oneCol] = 1;
			return zeroes;
        }

		public static Matrix[] BasisMatrices(int rows, int cols)
        {
			List<Matrix> basisMatrices = new(rows * cols);
			for (int oneRow = 0; oneRow < rows; oneRow++)
            {
				for (int oneCol = 0; oneCol < cols; oneCol++)
                {
					basisMatrices.Add(BasisMatrix(rows, cols, oneRow, oneCol));
                }
            }
			return basisMatrices.ToArray();
        }

		private static double epsilon = ScalarFunctions.Epsilon;

		public static Matrix NumericPartialDerivs(Func<Matrix, double> f, Matrix differentiateAt)
        {
			List<List<double>> partials = new();
			for (int row = 0; row < differentiateAt.RowCount; row++)
            {
				List<double> partialsForRow = new();
				for (int col = 0; col < differentiateAt.ColumnCount; col++)
                {
					Matrix direction = BasisMatrix(differentiateAt.RowCount, differentiateAt.ColumnCount, row, col);
					double difference = f(differentiateAt + epsilon * direction) - f(differentiateAt - epsilon * direction);
					partialsForRow.Add(difference / (2 * epsilon));
                }
				partials.Add(partialsForRow);
            }
			return _builder.DenseOfRows(partials);
        }
	}
}