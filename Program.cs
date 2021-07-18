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
            NeuralNetwork net = new(10, 10, 5);
            net.WriteToDirectory("../../../NeuralNetworkLearning/layers");
            net = new("../../../NeuralNetworkLearning/layers");

            Vector<double> input = Vector<double>.Build.Random(10);
            Console.WriteLine(net.Output(input));
        }
    }
}
