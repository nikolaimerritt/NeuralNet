using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace CatchTheCheese
{
    public class QLearner
    {
        private double learningRate;
        private double futureDiscount;

        private QTable qTable = new();
        private Random rng = new();


        public QLearner(double learningRate, double futureDiscount)
        {
            this.learningRate = learningRate;
            this.futureDiscount = futureDiscount;
        }


        private Move chooseMoveExploreExploit(Level level, double exploreProb)
        {

            if (rng.NextDouble() <= exploreProb)
            {
                return level.validMoves[rng.Next(level.validMoves.Length)];
            }
            else
            {
                return qTable.moveWithMaxQValue(level);
            }
        }


        private void displayLevel(Level level)
        {
            Console.Clear();
            Console.WriteLine(level);
            Thread.Sleep(500);
        }

        private void learnFromGame(double exploreProb)
        {
            Level level = new Level();
            bool gameOver = false;
            while (!gameOver)
            {
                Level levelBeforeMove = level.deepCopy();
                Move move = chooseMoveExploreExploit(levelBeforeMove, exploreProb);
                level.makeMove(move);
                qTable.updateEntry(levelBeforeMove, move, level, learningRate, futureDiscount);

                gameOver = level.gameStatus == GameStatus.LOSS || level.gameStatus == GameStatus.WIN;
            }
        }

        public void learnFromGames(int numGames)
        {
            for (int gameNum = 1; gameNum <= numGames; gameNum++)
            {
                Console.WriteLine(gameNum);
                learnFromGame(exploreProb: 0.1);
            }
        }



        public void demoGame()
        {
            Level level = new();
            bool gameOver = false;
            string header = "\n============================================\n";

            displayLevel(level);
            while (!gameOver)
            {
                level.makeMove(qTable.moveWithMaxQValue(level));
                gameOver = level.gameStatus == GameStatus.WIN || level.gameStatus == GameStatus.LOSS;
                displayLevel(level);
            }

            if (level.gameStatus == GameStatus.WIN)
            {
                Console.WriteLine($"{header}\tHe found the cheese!{header}");
            }
            else if (level.gameStatus == GameStatus.LOSS)
            {
                Console.WriteLine($"{header}\tNooooo he died :({header}");
            }
        }
    }
}
