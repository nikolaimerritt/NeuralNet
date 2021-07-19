using System;
using MathNet.Numerics.LinearAlgebra;
using CatchTheCheeseGame;
using QTableLearning;
using NeuralNetLearning;

namespace CatchTheCheeseGame
{
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
        static void Main(string[] args)
        {
            NeuralNetwork net = net = new("../../../NeuralNetworkLearning/layers");

            Vector<double> input = Vector<double>.Build.Random(10);
            Vector<double> desiredOutput = Vector<double>.Build.Random(5);
            net.StochasticGradientDescent(input, desiredOutput, NeuralLayerConfig.MSEderiv, 0.001);
        }
    }
}
