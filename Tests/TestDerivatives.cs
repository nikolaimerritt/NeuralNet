using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.Distributions;

using Maths;
using NeuralNetLearning;

namespace Tests
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;

    static class TestDerivatives
    {
        private static void Compare(double analytic, double approx)
        {
            double tolerance = 1e-7;
            double error = Math.Abs(approx - analytic);
            Console.WriteLine($"error: {error:E2}");
            if (error > tolerance)
            {
                throw new Exception($"\nError of {error:E2} is greater than the error tolerance of {tolerance:E2}");
            }
        }

        private static void Compare(Vector analytic, Vector approx)
        {
            if (analytic.Count != approx.Count)
            {
                throw new Exception($"Could not compare the analytically-obtained vector of length {analytic.Count} to the approximated vector of length {approx.Count}");
            }
            for (int i = 0; i < analytic.Count; i++)
            {
                Compare(analytic[i], approx[i]);
            }
        }

        private static void Compare(Matrix analytic, Matrix approx)
        {
            if (analytic.RowCount != approx.RowCount || analytic.ColumnCount != approx.ColumnCount)
            {
                throw new Exception($"Could not compare the analytically-obtained matrix with dimension ({analytic.RowCount}, {analytic.ColumnCount}) and the approximated matrix with dimension ({approx.RowCount}, {approx.ColumnCount})");
            }
            for (int row = 0; row < analytic.RowCount; row++)
            {
                for (int col = 0; col < analytic.ColumnCount; col++)
                {
                    Compare(analytic[row, col], approx[row, col]);
                }
            }
        }

        private static void TestScalarFunction(Func<double, double> f,  Func<double, double> analyticDeriv, string name)
        {
            Console.Write($"\nTesting {name}... \t\t");
            for (double x = -10; x <= 10; x++)
            {
                double analytic = analyticDeriv(x);
                double approx = ScalarFunctions.NumericDerivative(f, x);
                Compare(analytic, approx);
            }
            Console.WriteLine("\t\t... passed");
        }


        private static void TestVectorFunction(Func<Vector, double> f, Vector differentiateAt, Vector analyticGrad, string name)
        {
            Console.Write($"\nTesting {name}... \t\t");
            Vector approx = VectorFunctions.NumericPartialDerivs(f, differentiateAt);
            Compare(analyticGrad, approx);
            Console.WriteLine("\t\t...passed");
        }


        private static void TestMatrixFunction(Func<Matrix, double> f, Matrix differentiateAt, Matrix analyticGrad, string name)
        {
            Console.Write($"\nTesting {name} ... \t\t");
            Matrix approx = MatrixFunctions.NumericPartialDerivs(f, differentiateAt);
            Compare(analyticGrad, approx);
            Console.Write("\t\t... passed");
        }


        public static void TestScalarFunctions()
        {
            TestScalarFunction(ScalarFunctions.Relu, ScalarFunctions.ReluDeriv, "ReLU");
            TestScalarFunction(ScalarFunctions.Sigmoid, ScalarFunctions.SigmoidDeriv, "sigmoid");
        }

        private static void TestWeight(int numLayers, int layerIdx)
        {
            int[] layerSizes = Enumerable.Range(0, numLayers+1).Select(_ => 5).ToArray();
            NeuralNetwork blankNet = new (layerSizes);

            Matrix weight = blankNet._layers[layerIdx]._weight;
            blankNet.WriteToDirectory();

            Vector input = VectorFunctions.StdUniform(5);
            Vector desiredOutput = VectorFunctions.StdUniform(5);

            double costAsFnOfWeight(Matrix newWeight)
            {
                NeuralNetwork net = blankNet.DeepCopyWithModification(newWeight, layerIdx);
                return net.Cost(input, desiredOutput);
            }

            NeuralNetwork net = blankNet.DeepCopy();
            (Matrix[] analyticGrads, var _) = net.WeightsAndBiasesOfGradDescent(input, desiredOutput, learningRate: 0);

            TestMatrixFunction(costAsFnOfWeight, weight, analyticGrads[layerIdx], $"Cost as a function of weight {layerIdx+1}");
        }

        public static void TestBias(int numLayers, int layerIdx)
        {

            int[] layerSizes = Enumerable.Range(0, numLayers+1).Select(_ => 5).ToArray();
            
            NeuralNetwork blankNet = new(layerSizes);

            Vector bias = blankNet._layers[layerIdx]._bias;
            Vector input = VectorFunctions.StdUniform(5);
            Vector desiredOutput = VectorFunctions.StdUniform(5);

            double costAsFnOfBias(Vector newBias)
            {
                NeuralNetwork net = blankNet.DeepCopyWithModification(newBias, layerIdx);
                return net.Cost(input, desiredOutput);
            }

            NeuralNetwork net = blankNet.DeepCopy();
            (var _, Vector[] analyticGrads) = net.WeightsAndBiasesOfGradDescent(input, desiredOutput, learningRate: 0);

            TestVectorFunction(costAsFnOfBias, bias, analyticGrads[layerIdx], $"Cost as a function of bias {layerIdx+1}");
        }

        public static void TestWeightsAndBiases(int numLayers)
        {
            for (int layerIdx = numLayers - 1; layerIdx >= 0; layerIdx--)
            {
                Console.WriteLine($"Testing layer {layerIdx + 1}:");

                Console.Write($"\t\t Testing bias {layerIdx+1}... ");
                TestBias(numLayers, layerIdx);
                Console.WriteLine("... passed");

                Console.Write($"\t\t Testing weight {layerIdx+1}...");
                TestWeight(numLayers, layerIdx);
                Console.WriteLine("... passed");

            }
        }
    }
}
