using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Threading.Tasks;
using NeuralNetLearning.Maths;


namespace NeuralNetLearning
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;
    using static ParameterFactory;

    public static class NeuralNetFactory
    {
        public static NeuralNet RandomCustomisedForTanhSigmoid(IList<NeuralLayerConfig> layerConfigs, GradientDescender gradientDescender, CostFunction cost)
        {
            Parameter param = XavierInit(LayerSizesFromConfigs(layerConfigs));
            Activation[] activators = ActivationsFromConfigs(layerConfigs);

            return new NeuralNet(param, activators, gradientDescender, cost);
        }

        public static NeuralNet RandomCustomisedForRelu(IList<NeuralLayerConfig> layerConfigs, GradientDescender gradientDescender, CostFunction cost)
        {
            Parameter param = KaimingInit(LayerSizesFromConfigs(layerConfigs));
            Activation[] activations = ActivationsFromConfigs(layerConfigs);

            return new NeuralNet(param, activations, gradientDescender, cost);
        }

        public static NeuralNet RandomCustomisedForMiniBatch(IList<NeuralLayerConfig> layerConfigs, IEnumerable<(Vector, Vector)> firstMiniBatch, GradientDescender gradientDescender, CostFunction cost)
        {
            IList<Vector> miniBatchInputs = firstMiniBatch.Select(pair => pair.Item1).ToList();
            return RandomCustomisedForMiniBatch(layerConfigs, miniBatchInputs, gradientDescender, cost);
        }

        public static NeuralNet RandomCustomisedForMiniBatch(IList<NeuralLayerConfig> layerConfigs, IEnumerable<Vector> firstMiniBatch, GradientDescender gradientDescent, CostFunction cost)
        {
            Activation[] activators = ActivationsFromConfigs(layerConfigs);
            Parameter param = LSUVInit(LayerSizesFromConfigs(layerConfigs), activators, firstMiniBatch);
            
            return new NeuralNet(param, activators, gradientDescent, cost);
        }


        private static int[] LayerSizesFromConfigs(IList<NeuralLayerConfig> layerConfigs)
                => layerConfigs
                    .Select(l => l.LayerSize)
                    .ToArray();

        private static Activation[] ActivationsFromConfigs(IList<NeuralLayerConfig> layerConfigs)
        {
            if (!(layerConfigs.First() is InputLayer))
                throw new ArgumentException($"Expected the first layer to be of type {typeof(InputLayer)}");

            List<Activation> activators = new(layerConfigs.Count - 1);
            for (int i = 1; i < layerConfigs.Count - 1; i++)
            {
                if (!(layerConfigs[i] is HiddenLayer))
                    throw new ArgumentException($"Expected layer {i} to be of type {typeof(HiddenLayer)}");

                activators.Add((layerConfigs[i] as HiddenLayer).Activator);
            }

            if (!(layerConfigs.Last() is OutputLayer))
                throw new ArgumentException($"Expected the last layer to be of type {typeof(OutputLayer)}");

            activators.Add((layerConfigs.Last() as OutputLayer).Activator);

            return activators.ToArray();
        }
    }
}