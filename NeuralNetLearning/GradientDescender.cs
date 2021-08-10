using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using NeuralNetLearning.Maths;

namespace NeuralNetLearning
{
    public abstract class GradientDescender
    {
        public static readonly string NameFile = "gradient-descender-name.txt";
        
        public static readonly string hyperParamsFile = "hyper-params.txt";

        public abstract Parameter GradientDescentStep(Parameter gradient);

        public abstract void WriteToDirectory(string directoryPath);

        public static GradientDescender ReadFromDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                throw new DirectoryNotFoundException($"Could not find directory {directoryPath}");

            string typeName = HyperParamEncoder.DecodeTypeName($"{directoryPath}/{hyperParamsFile}");
            if (typeName == nameof(StochasticGradientDescender))
                return StochasticGradientDescender.Read(directoryPath);

            if (typeName == nameof(AdamGradientDescender))
                return AdamGradientDescender.Read(directoryPath);

            else throw new Exception($"Could not recognise the gradient descender with type name {typeName}");
        }
    }
}
