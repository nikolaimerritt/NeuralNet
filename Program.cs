using System;

namespace CatchTheCheese
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            QLearner learner = new QLearner(learningRate: 0.1, futureDiscount: 1, exploreProb: 0.3);
            learner.learnFromGames(numGames: 5000);

            Console.WriteLine("are you ready to see a demo?");
            learner.demoGame();
            Console.ReadKey();
        }
    }
}
