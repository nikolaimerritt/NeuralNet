using System;

namespace CatchTheCheese
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            QLearner learner = new QLearner(learningRate: 0.5, futureDiscount: 0.9);
            learner.learnFromGames(numGames: 100);

            Console.WriteLine("do you want to see the demo?");
            Console.ReadKey();

            learner.demoGame();
        }
    }
}
