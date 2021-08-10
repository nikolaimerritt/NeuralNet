using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetLearning.Maths
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;

    public static class VectorFunctions
    {
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
