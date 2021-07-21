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


		public static Matrix StdNormal(int inputDim, int outputDim)
			=> Matrix.Build.Random(outputDim, inputDim, _stdNormal);


		public static Matrix StdUniform(int inputDim, int outputDim) // [-1, 1]
			=> Matrix.Build.Random(outputDim, inputDim, _stdUniform);


		public static Matrix Read(string filepath)
			=> DelimitedReader.Read<double>(filepath, delimiter: "\t");


		public static void Write(this Matrix matrix, string filepath)
			=> DelimitedWriter.Write(filepath, matrix, delimiter: "\t");

	}
}
