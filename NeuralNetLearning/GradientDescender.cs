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
        /// <summary>
        /// The filename containing the name of the Gradient Descender when written to a directory.
        /// </summary>
        public static readonly string NameFile = "gradient-descender-name.txt";
        
        /// <summary>
        /// The filename containing the hyper-parameters of the Gradient Descender when written to a directory.
        /// </summary>
        public static readonly string hyperParamsFile = "hyper-params.txt";

        /// <summary>
        /// Returns the change to be added to parameter when executing gradient descent. For example, in stochastic gradient descent,
        /// <code>
        /// GradientDescentStep(Parameter gradient)
        ///     => -1 * _learningRate * gradient;
        /// </code>
        /// </summary>
        /// <param name="gradient"> The cost gradient </param>
        public abstract Parameter GradientDescentStep(Parameter gradient);

        /// <summary>
        /// Writes the Gradient Descender to the supplied directory. This is to be done so that 
        /// </summary>
        /// <param name="directoryPath"></param>
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
