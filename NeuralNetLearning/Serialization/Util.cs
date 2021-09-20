using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLearning.Serialization
{
    internal static class Util
    {
        public static object CreateInstanceOfType(Type t, IDictionary<string, object> namedArguments)
        {
            ConstructorInfo constructor = t
                .GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .Where(c => c.GetParameters().Length == namedArguments.Count)
                .Single();
            return constructor.Invoke(ReorderNamedArguments(constructor, namedArguments));
        }

        private static object[] ReorderNamedArguments(MethodBase method, IDictionary<string, object> namedArguments)
        {
            string[] argumentNamesInOrder = method.GetParameters().Select(p => p.Name).ToArray();
            return ReorderValues(namedArguments, argumentNamesInOrder);
        }

        /*
        public static Dictionary<string, string> SerializedNameToFieldName(Type gradientDescenderChild)
        {
            var serializedNameData = from fieldInfo in gradientDescenderChild.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                       let attribute = fieldInfo.GetCustomAttributes(typeof(SerializableHyperParameter), inherit: true)
                       where attribute.Length == 1
                       select (serializedName: (attribute.Single() as SerializableHyperParameter).Name,
                               fieldName: fieldInfo.Name);

            return serializedNameData.ToDictionary(s => s.serializedName, s => s.fieldName);
        } */

        private static T[] ReorderValues<T>(IDictionary<string, T> dictionary, string[] keysInCorrectOrder)
        {

            if (dictionary.Keys.Any(k => !keysInCorrectOrder.Contains(k)) || keysInCorrectOrder.Any(k => !dictionary.Keys.Contains(k)))
                throw new Exception($"The dictionary's keys does not match with the list of keys in correct order");

            return keysInCorrectOrder.Select(k => dictionary[k]).ToArray();
        }

        public static Type GetDerivedTypeWithName(Type parent, string name)
        {
            var derivedTypes = from type in Assembly.GetAssembly(parent).GetTypes()
                               where type.IsClass && !type.IsAbstract && type.IsSubclassOf(parent)
                               select type;

            return derivedTypes.Where(type => type.Name == name).Single();
        }

        public static object ReadIntBoolDouble(string value)
        {

            if (bool.TryParse(value, out bool boolResult))
                return boolResult;

            if (int.TryParse(value, out int intResult))
                return intResult;

            if (double.TryParse(value, out double doubleResult))
                return doubleResult;

            throw new ArgumentException($"The value {value} could not be parsed as a double, int or boolean");
        }
    }
}
