using System;
using System.Collections.Generic;
using System.Linq;

namespace CatchTheCheese
{
    public enum CellType
    {
        EMPTY = 1, BLOCK = 2, PLAYER = 3, ENEMY = 4, GOAL = 5
    }
    public record Cell
    {
        public CellType cellType { get; private set; }

        private static readonly Dictionary<CellType, string> cellTypeToStr = new Dictionary<CellType, string>()
        {
            [CellType.EMPTY] = "⬛",
            [CellType.BLOCK] = "🧱",
            [CellType.PLAYER] = "🐁",
            [CellType.ENEMY] = "💀",
            [CellType.GOAL] = "🧀"
        };

        public Cell(CellType cellType) 
            => this.cellType = cellType;

        public Cell(string str) 
            => this.cellType = cellTypeToStr.FirstOrDefault(pair => pair.Value == str).Key;

        public override string ToString() 
            => cellTypeToStr[this.cellType];

    }
}
