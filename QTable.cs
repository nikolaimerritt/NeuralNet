using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

using LookupRow = System.Collections.Generic.Dictionary<CatchTheCheese.Move, double>;
using LookupTable = System.Collections.Generic.Dictionary<CatchTheCheese.Level, System.Collections.Generic.Dictionary<CatchTheCheese.Move, double>>;

namespace CatchTheCheese
{
	public class QTable
	{
		private LookupTable lookupTable;
		private Random rng = new();

        private static Dictionary<GameStatus, double> rewards = new()
        {
            [GameStatus.LOSS] = -10,
            [GameStatus.NEUTRAL] = -0.01,
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
					lookupRow[move] = 10 * rng.NextDouble();
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


        public Move bestMove(Level level)
        {
            if (this.containsLevel(level))
            {
                LookupRow lookupRow = lookupTable[level];
                return lookupRow.Keys.Aggregate((mv1, mv2) => lookupRow[mv1] > lookupRow[mv2] ? mv1 : mv2);
            }
            throw new KeyNotFoundException($"QTable does not contain a row for level \n{level}");
        }

        private double maxFutureQVal(Level level)
        {
            double max = 0;
            LookupRow lookupRow;
            if (this.lookupTable.TryGetValue(level, out lookupRow))
            {
                foreach (Move possibleMove in lookupRow.Keys)
                {
                    Level futureLevel = level.deepCopy();
                    futureLevel.makeMove(possibleMove);
                    LookupRow futureLookupRow;
                    if (lookupTable.TryGetValue(futureLevel, out futureLookupRow))
                    {
                        double maxFutureQVal = futureLookupRow.Values.Max();
                        if (maxFutureQVal > max)
                        {
                            max = maxFutureQVal;
                        }
                    }
                }
            }
            return max;
        }

        private double updatedQValue(Level level, Move move, GameStatus response, double learningRate, double futureDiscount)
        {
            double oldQVal = 0;
            if (lookupTable.ContainsKey(level) && lookupTable[level].ContainsKey(move))
            {
                // should always happen as lookup table initialised with all levels and all valid moves
                oldQVal = lookupTable[level][move];
            }
            else
            {
                throw new KeyNotFoundException($"QTable does not have entry for level\n{level} with hash code {level.GetHashCode()}");
            }
            double reward = rewards[response];

            double oldQValWeighting = (1 - learningRate) * oldQVal;
            double futureQValWeighting = learningRate * (reward + futureDiscount * maxFutureQVal(level));
            return oldQValWeighting + futureQValWeighting;
        }

        public void updateEntry(Level level, Move move, GameStatus response, double learningRate, double futureDiscount)
        {
            if (lookupTable.ContainsKey(level) && lookupTable[level].ContainsKey(move))
            {
                double newQVal = updatedQValue(level, move, response, learningRate, futureDiscount);
                lookupTable[level][move] = newQVal;
            }
            else
            {
                // should never happen as lookup table initialised with all possible levels
                // not sure what to do about that
                throw new KeyNotFoundException($"QTable has no entry for level \n{level}");
                // lookupTable[level] = new LookupRow() { [move] = newQVal };
            }
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
