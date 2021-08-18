﻿using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using NeuralNetLearning.Maths;

namespace NeuralNetLearning
{
    using TrainingPair = Tuple<Vector<double>, Vector<double>>;

	public class NeuralNet
    {
        private Parameter _param;
        private readonly CostFunction _cost;
        private readonly Activation[] _activators;
        private readonly GradientDescender _gradientDescender;

        private static readonly string costFile = "cost.txt";
        private static readonly string paramsFolder = "parameters";
        private static readonly string activatorsFolder = "activators";
        private static readonly string gradDescenderFolder = "gradient-descender";
        private static readonly Random _rng = new();

        public int LayerCount
        {
            get => _param.LayerCount;
        }

        public int[] LayerSizes
        {
            get => _param.LayerSizes;
        }

        
        public NeuralNet(Parameter param, Activation[] activators, GradientDescender gradientDescent, CostFunction cost)
        {
            if (activators.Length != param.LayerCount)
                throw new ArgumentException($"Expected {param.LayerCount} activators, but received {activators.Length}");

            _param = param;
            _cost = cost;
            _activators = activators;
            _gradientDescender = gradientDescent;
        }

        protected Parameter AverageGradient(Parameter param, IReadOnlyList<TrainingPair> trainingPairs) // assuming that trainingPairs.Count() is not extremely large
        {
            Parameter total = ParameterFactory.Zero(param.LayerSizes);
            foreach ((Vector<double> input, Vector<double> desiredOutput) in trainingPairs)
                total.InPlaceAdd(_param.CostGrad(input, desiredOutput, _activators, _cost));

            total.InPlaceDivide(trainingPairs.Count);
            return total;
        }

        public void GradientDescent(TrainingPair[] trainingPairs, int batchSize, int numEpochs = 5)
        {
            ArraySegment<TrainingPair> trainingPairSpan = new(trainingPairs);
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                Shuffle(trainingPairs);
                for (int batchIdx = 0; batchIdx < trainingPairs.Length; batchIdx += batchSize)
                {
                    int size = Math.Min(batchSize, trainingPairs.Length - batchIdx);
                    ArraySegment<TrainingPair> trainingBatch = trainingPairSpan.Slice(batchIdx, size);
                    Parameter update = _gradientDescender.GradientDescentStep(AverageGradient(_param, trainingBatch));
                    _param.InPlaceAdd(update);
                }
                Console.WriteLine($"Epoch {epoch} / {numEpochs} \t \t Avg training cost: {AverageCost(trainingPairs)}");
            }
        }

        public void WriteToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            _param.WriteToDirectory($"{directoryPath}/{paramsFolder}");
            WriteActivatorsToDirectory($"{directoryPath}/{activatorsFolder}");
            _gradientDescender.WriteToDirectory($"{directoryPath}/{gradDescenderFolder}");
            _cost.WriteToFile($"{directoryPath}/{costFile}");
        }

        public static NeuralNet ReadFromDirectory(string directoryPath)
        {
            Parameter param = Parameter.ReadFromDirectory($"{directoryPath}/{paramsFolder}");
            Activation[] activators = ReadActivationsFromDirectory($"{directoryPath}/{activatorsFolder}");
            GradientDescender gradientDescender = GradientDescender.ReadFromDirectory($"{directoryPath}/{gradDescenderFolder}");
            CostFunction cost = CostFunction.ReadFromFile($"{directoryPath}/{costFile}");

            return new NeuralNet(param, activators, gradientDescender, cost);
        }

        private static Activation[] ReadActivationsFromDirectory(string directory)
        {
            if (!Directory.Exists(directory))
                throw new FileNotFoundException($"Could not find directory {directory}");

            List<string> activationFiles = Directory.GetFiles(directory).ToList();
            activationFiles.Sort();

            return activationFiles.Select(Activation.ReadFromFile).ToArray();
        }

        private void WriteActivatorsToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            for (int i = 0; i < _activators.Length; i++)
                _activators[i].WriteToFile($"{directoryPath}/activator {i + 1}.txt");
        }

        public Vector<double> Output(Vector<double> input)
            => _param.Output(input, _activators);

        public double AverageCost(TrainingPair[] trainingPairs)
            => trainingPairs
            .Select(pair => _cost.Apply(Output(pair.Item1), pair.Item2))
            .Average();

        private static void Shuffle<T>(IList<T> list)
        {
            int swapIdx = list.Count;
            while (swapIdx > 1)
            {
                swapIdx--;
                int replaceIdx = _rng.Next(swapIdx + 1);
                T value = list[replaceIdx];
                list[replaceIdx] = list[swapIdx];
                list[swapIdx] = value;
            }
        }
    }
}