using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.Distributions;

namespace NeuralNetLearning.Maths
{
	using Matrix = Matrix<double>;
    public static class MatrixFunctions
    {
		private static readonly Normal _gaussianDist = new (mean: 0, stddev: 1);
		private static readonly ContinuousUniform _stdNormalDist = new (lower: -1, upper: 1);

		public static Matrix StdNormal(int rows, int cols)
			=> DenseMatrix.CreateRandom(rows, cols, _gaussianDist);


		public static Matrix StdUniform(int rows, int cols) // [-1, 1]
			=> DenseMatrix.CreateRandom(rows, cols, _stdNormalDist);

		public static Matrix GaussianOrthonormal(int rows, int cols)
        {
			Matrix gaussian = DenseMatrix.CreateRandom(rows, cols, _gaussianDist);
			Svd<double> svdSolver = gaussian.Svd();
			Matrix rowByRowOrthonormal = svdSolver.U;
			Matrix colByColOrthonormal = svdSolver.VT;

			Matrix biggest = rows > cols ? rowByRowOrthonormal : colByColOrthonormal;
			return SubMatrix(biggest, rows, cols);
        }

		public static Matrix Read(string filepath)
			=> DelimitedReader.Read<double>(filepath, delimiter: "\t");


		public static void Write(this Matrix matrix, string filepath)
			=> DelimitedWriter.Write(filepath, matrix, delimiter: "\t");

		private static Matrix SubMatrix(Matrix bigMatrix, int maxRows, int maxCols)
        {
			var columns = bigMatrix
				.EnumerateColumns()
				.ToList()
				.GetRange(0, maxCols)
				.Select(bigCol => bigCol.SubVector(0, maxRows));
			
			return Matrix.Build.DenseOfColumnVectors(columns);
        }

		public static Matrix BasisMatrix(int rows, int cols, int oneRow, int oneCol)
        {
			Matrix zeroes = new DenseMatrix(rows, cols);
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
	}
}