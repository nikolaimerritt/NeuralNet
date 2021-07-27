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


        static void Main(string[] args)
        {
            TestDerivatives.TestWeightsAndBiases(3);
        }
    }
}
