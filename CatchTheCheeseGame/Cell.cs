using System;
using System.Collections.Generic;
using System.Linq;

namespace CatchTheCheeseGame
{
    public enum CellType
    {
        EMPTY, BLOCK, ENEMY, PLAYER, GOAL
    }
    public record Cell
    {
        public CellType cellType { get; private set; }

        private static readonly Dictionary<CellType, string> cellTypeToStr = new Dictionary<CellType, string>()
        {
            [CellType.EMPTY]  = "⬛",
            [CellType.BLOCK]  = "🧱",
            [CellType.PLAYER] = "🐁",
            [CellType.ENEMY]  = "💀",
            [CellType.GOAL]   = "🧀"
        };

        public Cell(CellType cellType) 
            => this.cellType = cellType;

        public Cell(string str) 
            => this.cellType = cellTypeToStr.FirstOrDefault(pair => pair.Value == str).Key;

        public override string ToString() 
            => cellTypeToStr[this.cellType];

    }
}
