using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using CatchTheCheeseGame;
using QTableLearning;
using NeuralNetLearning;

namespace CatchTheCheeseGame
{
    using Vector = Vector<double>;
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
            Vector input = NeuralLayer.StdUniformVector(1);
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
                    net.StochasticGradientDescent(input, desiredOutput, NeuralLayerConfig.MSEderiv, learningRate: 10e-2);
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
                costs.Add(NeuralLayerConfig.MSE(output, desiredOutput));
            }
            return costs.Average();
        }

        static void Main(string[] args)
        {
            NeuralNetwork net = new(1, 8, 8, 1);
            TrainNet(net, numTrainingPairs: (int) 10e4, numEpochs: 100, numTestingPairs: (int) 10e2);

            List<(Vector, Vector)> finalTests = RandomTrainingPairs(10);
            Console.WriteLine("\n\n\n===================== demo =====================");
            foreach ((Vector input, Vector desiredOutput) in finalTests)
            {
                double output = net.Output(input)[0];
                Console.WriteLine($"{input[0]:0.###} \t --> \t {output:0.###} \t \t (correct answer was {desiredOutput[0]:0.###})");
            }

            net.WriteToDirectory("../../../NeuralNetworkLearning/");
        }
    }
}
