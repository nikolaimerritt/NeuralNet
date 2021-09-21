using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetLearning.Serialization;
using System;
using System.IO;
using System.Linq;

namespace NeuralNetLearning.Maths.Activations
{
    using Matrix = Matrix<double>;
    using Vector = Vector<double>;

    /// <summary>
    /// Represents the activation function that is applied to a neural layer.
    /// </summary>
    public abstract class Activation
    {
        /// <summary>
        /// Returns the resulting <see cref="Vector{Double}"/> on applying the activation function to <paramref name="input"/>.
        /// </summary>
        public abstract Vector Apply(Vector input);

        /// <summary>
        /// Returns the matrix of derivatives of the activation function evaluated with <paramref name="input"/>. 
        /// <para>
        /// The <c>[r, c]</c> entry is the derivative of the <c>r</c>th component of the activation function, with respect to the <c>c</c>th component of the input.
        /// </para>
        /// </summary>
        public abstract Matrix ApplyDerivative(Vector input);

        /// <summary>
        /// Writes the name of the activation function and its hyper-parameters to <paramref name="filepath"/>.
        /// </summary>
        //public virtual void WriteToFile(string filepath)
          //  => HyperParamEncoder.EncodeToFile(this.GetType().Name, filepath);

        /// <summary>
        /// Reads the appropriate sub-class of <see cref="Activation"/> that was written to <paramref name="filepath"/>.
        /// <para>
        /// The returned sub-class of <see cref="Activation"/> has the same type and hyper-parameter vales as the sub-class of <see cref="Activation"/> that was written.
        /// </para>
        /// </summary>
        /* public static Activation ReadFromFile(string filepath)
        {
            if (!File.Exists(filepath))
                throw new FileNotFoundException($"Could not find file {filepath}");

            string typeName = HyperParamEncoder.DecodeTypeName(filepath);
            if (typeName == nameof(ReluActivation))
                return new ReluActivation(HyperParamEncoder.Decode(filepath, "leak"));

            if (typeName == nameof(TanhActivation))
                return new TanhActivation();

            if (typeName == nameof(IdentityActivation))
                return new IdentityActivation();

            else throw new Exception($"Could not recognise activation with type name {typeName}");
        } */
    }

    

    

    

    
}
