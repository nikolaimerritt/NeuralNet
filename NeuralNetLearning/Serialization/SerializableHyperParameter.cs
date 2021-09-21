using System;

namespace NeuralNetLearning.Serialization
{
    [AttributeUsage(validOn: AttributeTargets.Field, AllowMultiple = false)]
    public class SerializableHyperParameter : Attribute
    {
        public readonly string Name;

        public SerializableHyperParameter(string name)
        {
            Name = name;
        }
    }
}
