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
        private double propRandomGames;
        private double exploreProb;

        private QTable qTable = new();
        private Random rng = new();


        public QLearner(double learningRate, double futureDiscount, double exploreProb)
        {
            this.learningRate = learningRate;
            this.futureDiscount = futureDiscount;
            this.exploreProb = exploreProb;
        }


        private Move chooseRandomMove(Level level)
        {
            return level.validMoves[rng.Next(level.validMoves.Length)];
        }


        private Move chooseMoveExploreExploit(Level level)
        {
            if (rng.NextDouble() <= exploreProb)
            {
                return chooseRandomMove(level);
            }
            else
            {
                return qTable.bestMove(level);
            }
        }

        private Move readMoveFromUser(Level level)
        {
            switch (Console.ReadKey().Key)
            {
                case ConsoleKey.A or ConsoleKey.LeftArrow:
                    return Move.BACKWARDS;
                default:
                    return Move.FORWARDS;
            }
        }


        private void learnFromGame(Func<Level, Move> chooseMove)
        {
            Level level = new Level();
            bool gameOver = false;
            while (!gameOver)
            {
                Level levelBeforeMove = level.deepCopy();
                Move move = chooseMove(levelBeforeMove);
                level.makeMove(move);
                qTable.updateEntry(levelBeforeMove, move, level.gameStatus, learningRate, futureDiscount);

                gameOver = level.gameStatus == GameStatus.LOSS || level.gameStatus == GameStatus.WIN;
            }
        }

        public void learnFromGames(int numGames)
        {
            int numRandomGames = (int)Math.Round(0.3 * numGames);
            this.exploreProb = 0.9;
            for (int gameNum = 1; gameNum <= numRandomGames; gameNum++)
            {
                Console.WriteLine(gameNum);
                learnFromGame(chooseMoveExploreExploit);
            }
            this.exploreProb = 0.1;
            for (int gameNum = numRandomGames + 1; gameNum <= numGames; gameNum++)
            {
                Console.WriteLine(gameNum);
                learnFromGame(chooseMoveExploreExploit);
            }
        }

        public void demoGame()
        {
            Level level = new();
            while (level.gameStatus != GameStatus.WIN && level.gameStatus != GameStatus.LOSS)
            {
                Console.WriteLine(level);
                Console.WriteLine(qTable.levelDataAsString(level));
                Console.ReadKey();
                level.makeMove(qTable.bestMove(level));
            }
        }
    }
}
