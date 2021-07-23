using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using CatchTheCheeseGame;
using QTableLearning;
using NeuralNetLearning;
using Maths;
using Tests;

namespace CatchTheCheeseGame
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;
    class Program
    {
        static void CatchTheCheeseQTableDemo(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            QLearner learner = new QLearner(learningRate: 0.2, futureDiscount: 0.9);
            learner.learnFromGames(numGames: args.Length == 0 ? 1000 : int.Parse(args[0]));

            Console.WriteLine("do you want to see the demo?");
            Console.ReadKey();

            learner.demoGame();
        }

        static (Vector, Vector) RandomTrainingPair()
        {
            Vector input = 1 * VectorFunctions.StdUniform(1);
            Vector desiredOutput = Vector.Build.DenseOfArray(new double[] { input[0] * input[0] });
            return (input, desiredOutput);
        }

        static List<(Vector, Vector)> RandomTrainingPairs(int amount)
            =>  Enumerable
                .Range(0, amount)
                .Select(i => RandomTrainingPair())
                .ToList();

        static void TrainNet(in NeuralNetwork net, int numTrainingPairs, int numEpochs, int numTestingPairs)
        {
            List<(Vector, Vector)> testingPairs = RandomTrainingPairs(numTestingPairs);
            for (int epoch = 1; epoch <= numEpochs; epoch++)
            {
                double meanCost = MeanCost(net, testingPairs);
                Console.WriteLine($"Mean test cost at epoch {epoch} / {numEpochs}: \t {meanCost:0.#####}");

                foreach ((Vector input, Vector desiredOutput) in RandomTrainingPairs(numTrainingPairs))
                {
                    net.WeightsAndBiasesOfGradDescent(input, desiredOutput, learningRate: 10e-2);
                }
            }
            net.WriteToDirectory("../../../NeuralNetworkLearning/layers");
        }

        public static double MeanCost(NeuralNetwork net, List<(Vector, Vector)> trainingPairs)
        {
            List<double> costs = new();
            foreach ((Vector input, Vector desiredOutput) in trainingPairs)
            {
                Vector output = net.Output(input);
                costs.Add(VectorFunctions.MSE(output, desiredOutput));
            }
            return costs.Average();
        }

        static void Main(string[] args)
        {
            TestDerivatives.TestWeightsAndBiases(3);
        }
    }
}
