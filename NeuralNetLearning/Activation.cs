using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using NeuralNetLearning.Maths;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetLearning
{
    using Vector = Vector<double>;
    using Matrix = Matrix<double>;

    public abstract class Activation 
    {
        public abstract Vector Apply(Vector input);
        public abstract Matrix ApplyDerivative(Vector input);
        public virtual void WriteToFile(string filepath)
            => HyperParamEncoder.EncodeToFile(this.GetType().Name, filepath);

        public static Activation ReadFromFile(string filepath)
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
        }
    }

    public class ReluActivation : Activation
    {
        private readonly double _leak;
        public ReluActivation(double leak = 0)
        {
            _leak = leak;
        }

        public override Vector Apply(Vector<double> input)
        {
            double[] activation = new double[input.Count];
            for (int i = 0; i < activation.Length; i++)
                activation[i] = input[i] > 0 ? input[i] : _leak * input[i];

            return new DenseVector(activation);
        }

        public override Matrix ApplyDerivative(Vector<double> input)
        {
            double[] deriv = new double[input.Count];
            for (int i = 0; i < deriv.Length; i++)
                if (input[i] > 0)
                    deriv[i] = 1;
                else if (input[i] == 0)
                    deriv[i] = (1 + _leak) / 2;
                else
                    deriv[i] = _leak;

            return new DiagonalMatrix(deriv.Length, deriv.Length, deriv);
        }

        public override void WriteToFile(string filepath)
            => HyperParamEncoder.EncodeToFile(this.GetType().Name, filepath, ("leak", _leak));
    }

    public class TanhActivation : Activation
    {
        private Vector _cachedInput;
        private Vector _cachedTanh;
        public TanhActivation() { }

        public override Vector Apply(Vector input)
        {
            double[] activation = new double[input.Count];
            for (int i = 0; i < input.Count; i++)
                activation[i] = Math.Tanh(input[i]);

            _cachedInput = input;
            _cachedTanh = new DenseVector(activation);
            return _cachedTanh;
        }

        public override Matrix ApplyDerivative(Vector input)
        {
            double[] derivEntries = new double[input.Count];
            Vector tanh = input == _cachedInput ? _cachedTanh : input.PointwiseTanh();

            for (int i = 0; i < derivEntries.Length; i++)
                derivEntries[i] = 1 - Math.Pow(tanh[i], 2);

            return new DiagonalMatrix(derivEntries.Length, derivEntries.Length, derivEntries);
        }
    }

    public class IdentityActivation : Activation
    {
        public IdentityActivation() { }

        public override Vector<double> Apply(Vector<double> input)
            => input;

        public override Matrix ApplyDerivative(Vector<double> input)
            => Matrix.Build.DenseIdentity(input.Count);
    }

    public class SoftmaxActivation : Activation
    {
        private Vector _cachedInput;
        private Vector _cachedSoftmax;

        public SoftmaxActivation() { }

        public override Vector Apply(Vector input)
        {
            double maxInput = input.Max();
            double[] exponentials = new double[input.Count];
            for (int i = 0; i < exponentials.Length; i++)
                exponentials[i] = Math.Exp(input[i] - maxInput);

            _cachedInput = input;
            _cachedSoftmax = new DenseVector(exponentials) / exponentials.Sum();
            return _cachedSoftmax;
        }

        public override Matrix ApplyDerivative(Vector input)
        {
            Vector softmax = input == _cachedInput ? _cachedSoftmax : Apply(input);
            DiagonalMatrix softmaxAlongDiagonal = new(softmax.Count, softmax.Count, softmax.ToArray());
            return -Vector.OuterProduct(softmax, softmax) + softmaxAlongDiagonal;
        }
    }
}
