using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using CatchTheCheeseGame;

namespace QTableLearning
{
    using LookupRow = Dictionary<Move, double>;
    using LookupTable = Dictionary<Level, Dictionary<Move, double>>;

    public class QTable
	{
		private LookupTable lookupTable;
		private Random rng = new();

        private static Dictionary<GameStatus, double> rewards = new()
        {
            [GameStatus.LOSS] = -10,
            [GameStatus.NEUTRAL] = 0,
            [GameStatus.WIN] = 10
        };

        public QTable()
            => this.lookupTable = generateRandomTable();

        private LookupTable generateRandomTable()
        {
            LookupTable lookupTable = new();
			foreach (Level level in Level.allPossibleLevels)
            {
                LookupRow lookupRow = new();
				foreach (Move move in level.validMoves)
                {
					lookupRow[move] = rng.NextDouble();
                }
				lookupTable[level] = lookupRow;
            }
			return lookupTable;
        }

        public bool containsLevel(Level level)
            => lookupTable.ContainsKey(level);


        public string levelDataAsString(Level level)
        {
            if (this.containsLevel(level))
            {
                return lookupRowToStringBuilder(lookupTable[level]).ToString();
            }
            throw new KeyNotFoundException($"QTable does not contain a row for level \n{level}");
        }


        public Move moveWithMaxQValue(Level level)
        {
            LookupRow lookupRow = lookupTable[level];
            return lookupRow.Keys.Aggregate((move1, move2) => lookupRow[move1] > lookupRow[move2] ? move1 : move2);
        }

        private double maxQValue(Level level)
            => lookupTable[level][moveWithMaxQValue(level)];

        public void updateEntry(Level levelBeforeMove, Move move, Level levelAfterMove, double learningRate, double futureDiscount)
        {
            double oldValue = lookupTable[levelBeforeMove][move];
            double reward = rewards[levelAfterMove.gameStatus];
            double learnedValue = reward + futureDiscount * maxQValue(levelAfterMove);

            double newValue = (1 - learningRate) * oldValue + learningRate * learnedValue;

            lookupTable[levelBeforeMove][move] = newValue;
        }

        #region YUCKY_BOILERPLATE
        private StringBuilder lookupRowToStringBuilder(LookupRow lookupRow)
        {
            StringBuilder stringy = new();
            List<Move> moves = lookupRow.Keys.ToList();
            moves.Sort((mv1, mv2) => mv1.ToString().CompareTo(mv2.ToString()));
            
            foreach (Move move in moves)
            {
                string qValAsStr = lookupRow[move].ToString("0.0000");
                stringy.Append($"{move}:\t {qValAsStr}\t\t");
            }
            
            stringy.Append('\n');
            return stringy;
        }

        public override string ToString()
        {
            string header = "==============================\n";
            StringBuilder stringy = new(header);
            foreach (Level level in this.lookupTable.Keys)
            {
                stringy.Append(level);
                stringy.Append(lookupRowToStringBuilder(lookupTable[level]));
            }
            stringy.Append(header);
            return stringy.ToString();

        }
        #endregion YUCKY_BOILERPLATE
    }
}
