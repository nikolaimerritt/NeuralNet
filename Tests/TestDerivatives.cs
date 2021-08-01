using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Maths;
using NeuralNetLearning;

namespace Tests
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;

    static class TestDerivatives
    {
        private static Activation[] DefaultActivators(Parameter parameters)
            => Enumerable
                .Range(0, parameters.LayerCount)
                .Select(_ => Activation.TanhSigmoid)
                .ToArray();

        private static void Compare(double analytic, double numeric)
        {
            double tolerance = 1e-6;
            double error = Math.Abs(numeric - analytic);
            Console.WriteLine($"error: {error:E2}");
            if (error > tolerance)
            {
                throw new Exception($"Error of {error:E2} is greater than the error tolerance of {tolerance:E2}");
            }
        }

        private static double NumericCostGrad(Vector input, Vector desiredOutput, Parameter param, Parameter basisParam)
        {
            double epsilon = 1e-7;
            double cost(Parameter param)
                => param.Cost(input, desiredOutput, DefaultActivators(param));

            double difference = cost(param + epsilon * basisParam) - cost(param - epsilon * basisParam);
            return difference / (2 * epsilon); // approximation has error of order cost''(x) * epsilon^2 ~ 1e-14
        }



        private static Matrix GetWeight(Parameter param, int layerIdx)
        {
            PrivateObject privateParam = new (param);
            return (Matrix) privateParam.GetArrayElement("_weights", layerIdx);
        }

        private static void SetWeight(Parameter param, int layerIdx, Matrix weight)
        {
            PrivateObject privateParam = new(param);
            privateParam.SetArrayElement("_weights", weight, layerIdx);
        }

        private static Vector GetBias(Parameter param, int layerIdx)
        {
            PrivateObject privateParam = new(param);
            return (Vector) privateParam.GetArrayElement("_biases", layerIdx);
        }

        private static void SetBias(Parameter param, int layerIdx, Vector bias)
        {
            PrivateObject privateParam = new(param);
            privateParam.SetArrayElement("_biases", bias, layerIdx);
        }

        private static void TestWeight(int numLayers, int layerIdx)
        {
            int[] layerSizes = Enumerable.Range(0, numLayers+1).Select(i => numLayers + 10 - i).ToArray();
            Parameter param = Parameter.StdUniform(layerSizes);
            Vector input = VectorFunctions.StdUniform(layerSizes.First());
            Vector desiredOutput = VectorFunctions.StdUniform(layerSizes.Last());

            int cols = layerSizes[layerIdx];
            int rows = layerSizes[layerIdx + 1];

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    Matrix basisMatrix = MatrixFunctions.BasisMatrix(rows, cols, r, c);
                    Parameter basisParam = 0 * param;
                    SetWeight(basisParam, layerIdx, basisMatrix);
                    Parameter analyticGrad = param.CostGrad(input, desiredOutput, DefaultActivators(param));
                    
                    double analyticEntry = GetWeight(analyticGrad, layerIdx)[r, c];
                    double numeric = NumericCostGrad(input, desiredOutput, param, basisParam);

                    Compare(analyticEntry, numeric);
                }
            }
        }

        public static void TestBias(int numLayers, int layerIdx)
        {
            int[] layerSizes = Enumerable.Range(0, numLayers + 1).Select(i => numLayers + 10 - i).ToArray();
            Parameter param = Parameter.StdUniform(layerSizes);
            Vector input = VectorFunctions.StdUniform(layerSizes.First());
            Vector desiredOutput = VectorFunctions.StdUniform(layerSizes.Last());

            int length = layerSizes[layerIdx + 1];

            for (int i = 0; i < length; i++)
            {
                Vector basisVector = VectorFunctions.BasisVector(length, oneIdx: i);
                Parameter basisParam = 0 * param;
                SetBias(basisParam, layerIdx, basisVector);
                Parameter analyticGrad = param.CostGrad(input, desiredOutput, DefaultActivators(param));

                double analyticEntry = GetBias(analyticGrad, layerIdx)[i];
                double numeric = NumericCostGrad(input, desiredOutput, param, basisParam);

                Compare(analyticEntry, numeric);
            }
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
