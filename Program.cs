using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetLearning;
using Maths;
using Tests;

namespace CatchTheCheeseGame
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;
    using TrainingPairs = List<(Vector<double>, Vector<double>)>;

    class Program
    {
        static readonly Random rng = new();
        static TrainingPairs GetTrainingPairs(int amount)
        {
            TrainingPairs trainingPairs = new();
            for (int i = 0; i < amount; i++)
            {
                double x = 50 * 2*(rng.NextDouble() - 0.5);
                Vector input = Vector.Build.Dense(1, x);
                Vector desiredOutput = Vector.Build.Dense(1, x * x);
                trainingPairs.Add((input, desiredOutput));
            }
            return trainingPairs;
        }

        static void Demo(NeuralNet net, TrainingPairs trainingPairs)
        {
            foreach ((Vector input, Vector desiredOutput) in trainingPairs)
            {
                Console.WriteLine($"{input[0]:0.###} --> {net.Output(input)[0]:0.###} \t\t (correct: {desiredOutput[0]:0.###})");
            }
        }

        static void Main()
        {
            int[] layerSizes = new []
            {
                2, 3, 4
            };
            Activation[] activators = new[]
            {
                Activation.Relu, Activation.Identity
            };

            // NeuralNet net = new AdamNeuralNet(layerSizes, activators);
            // net.WriteToDirectory("../../../parameters");
            NeuralNet net = new AdamNeuralNet("../../../parameters");
            net.WriteToDirectory("../../../parameters-TEST");
        }
    }
}
