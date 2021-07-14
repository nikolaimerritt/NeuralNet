using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.VisualBasic.FileIO;

namespace CatchTheCheese
{
    public enum Move
    {
        FORWARDS, BACKWARDS, UP, DOWN
    }

    public enum GameStatus
    {
        WIN, LOSS, NEUTRAL
    }
    public class Level : IEquatable<Level>
    {
        private Cell[][] grid { get; set; }
        public Move[] validMoves
        {
            get
            {
                return Enum.GetValues(typeof(Move))
                    .Cast<Move>()
                    .Where(mv => isValidPlayerCoord(playerCoordAfterMove(mv)))
                    .ToArray();
            }
        }

        public GameStatus gameStatus
        {
            get
            {
                if (playerIsTouchingCell(new Cell(CellType.GOAL)))
                {
                    return GameStatus.WIN;
                }
                if (playerIsTouchingCell(new Cell(CellType.ENEMY)))
                {
                    return GameStatus.LOSS;
                }
                return GameStatus.NEUTRAL;
            }
        }

 
        public Level(string filepath)
        {
            this.grid = loadLevelGrid(filepath);
        }  

        public Level()
        {
            this.grid = loadLevelGrid(("../../../level.csv"));
        }


        public Level(Cell[][] grid)
        {
            this.grid = grid;
        }

        public static List<Level> allPossibleLevels
        {
            get
            {
                Level level = new Level();
                List<Level> levels = new List<Level> { level };
                for (int y = 0; y < level.grid.Length; y++)
                {
                    for (int x = 0; x < level.grid[y].Length; x++)
                    {
                        if (level.grid[y][x].cellType == CellType.EMPTY)
                        {
                            Level newLevel = level.deepCopy();
                            newLevel.movePlayerToCoord(new Coord(x, y));
                            levels.Add(newLevel);
                        }
                    }
                }
                return levels;
            }
        }

        private Cell[][] loadLevelGrid(string filepath)
        {
            List<Cell[]> level = new List<Cell[]>();
            using (TextFieldParser csvParser = new TextFieldParser(filepath))
            {
                csvParser.SetDelimiters(new string[] { "," });
                while (!csvParser.EndOfData)
                {
                    string[] fields = csvParser.ReadFields();
                    Cell[] row = Array.ConvertAll(fields, field => new Cell(field)).ToArray();
                    level.Add(row);
                }
            }
            return level.ToArray();
        }

        public Coord playerCoordFromGrid()
        {
            for (int y = 0; y < grid.Length; y++)
            {
                for (int x = 0; x < grid[y].Length; x++)
                {
                    if (grid[y][x].cellType == CellType.PLAYER)
                    {
                        return new Coord(x, y);
                    }
                }
            }
            return null;
            
        }

        private Coord playerCoordAfterMove(Move move)
        {
            Coord playerCoord = playerCoordFromGrid();
            switch (move)
            {
                case Move.BACKWARDS:
                    return playerCoord - new Coord(1, 0);
                case Move.FORWARDS:
                    return playerCoord + new Coord(1, 0);
                case Move.UP:
                    return playerCoord - new Coord(0, 1);
                case Move.DOWN:
                    return playerCoord + new Coord(0, 1);
            }
            return playerCoord;
        }

        private bool isValidPlayerCoord(Coord playerCoord)
        {
            if (playerCoord.y < 0 || playerCoord.y >= grid.Length || playerCoord.x < 0 || playerCoord.x >= grid[playerCoord.y].Length)
            {
                return false;
            }
            if (grid[playerCoord.y][playerCoord.x].cellType != CellType.EMPTY)
            {
                return false;
            }
            return true;
        }


        private void movePlayerToCoord(Coord newPlayerCoord)
        {
            Coord playerCoord = playerCoordFromGrid();
            grid[playerCoord.y][playerCoord.x] = new Cell(CellType.EMPTY);
            grid[newPlayerCoord.y][newPlayerCoord.x] = new Cell(CellType.PLAYER);
        }

        public void makeMove(Move move)
        {
            if (!validMoves.Contains(move))
            {
                throw new Exception($"Move {move} is invalid in level \n{this}\n with hash code {this.GetHashCode()}");
            }
            movePlayerToCoord(playerCoordAfterMove(move));
        }


        private bool playerIsTouchingCell(Cell cell)
        {
            Coord playerCoord = playerCoordFromGrid();
            for (int y = Math.Max(0, playerCoord.y - 1); y <= Math.Min(grid.Length - 1, playerCoord.y + 1); y++)
            {
                for (int x = Math.Max(0, playerCoord.x - 1); x <= Math.Min(grid[y].Length - 1, playerCoord.x + 1); x++)
                {
                    if (grid[y][x] == cell)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        #region YUCKY_BOILERPLATE
        public override string ToString()
        {
            string stringy = "";
            foreach (Cell[] row in this.grid)
            {
                stringy += "\n";
                foreach (Cell cell in row)
                {
                    stringy += $"{cell} ";
                }
            }
            return stringy;
        }


        public bool Equals(Level other)
        {
            if (other is null)
            {
                return false;
            }

            if (Object.ReferenceEquals(this, other))
            {
                return true;
            }

            if (this.GetType() != other.GetType())
            {
                return false;
            }

            Cell[] thisFlattenedGrid = this.grid.SelectMany(x => x).ToArray();
            Cell[] otherFlattenedGrid = other.grid.SelectMany(x => x).ToArray();

            if (thisFlattenedGrid.Length != otherFlattenedGrid.Length)
            {
                return false;
            }
            for (int i = 0; i < thisFlattenedGrid.Length; i++)
            {
                if (thisFlattenedGrid[i] != otherFlattenedGrid[i])
                {
                    return false;
                }
            }
            return true;
        }

        public override bool Equals(object obj)
            => this.Equals(obj as Level);


        public static bool operator==(Level lhs, Level rhs)
        {
            if (lhs is null && rhs is null)
            {
                return true;
            }
            else if (lhs is null || rhs is null)
            {
                return false;
            }

            return lhs.Equals(rhs);
        }

        public static bool operator !=(Level lhs, Level rhs)
            => !(lhs == rhs);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 19;
                Cell[] flattenedGrid = this.grid.SelectMany(x => x).ToArray();
                foreach (Cell cell in flattenedGrid)
                {
                    hash = 31 * hash + cell.GetHashCode();
                }
                return hash;
            }
        }

        public Level deepCopy()
        {
            List<Cell[]> newGrid = new List<Cell[]> { };
            foreach (Cell[] row in this.grid)
            {
                Cell[] newRow = new Cell[row.Length];
                for (int i = 0; i < row.Length; i++)
                {
                    newRow[i] = new Cell(row[i].cellType);
                }
                newGrid.Add(newRow);
            }
            return new Level(newGrid.ToArray());
        }
    }
    #endregion YUCKY_BOILERPLATE
}
