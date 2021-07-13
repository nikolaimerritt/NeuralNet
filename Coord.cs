using System;
using System.Collections.Generic;
using System.Text;

namespace CatchTheCheese
{
    public record Coord
    {
        public int x;
        public int y;

        public Coord(int x, int y)
        {
            this.x = x;
            this.y = y;
        }

        public static Coord operator +(Coord fstCoord, Coord sndCoord)
            => new Coord(fstCoord.x + sndCoord.x, fstCoord.y + sndCoord.y);

        public override string ToString()
            => $"({this.x}, {this.y})";

        public static Coord operator -(Coord fstCoord, Coord sndCoord)
            => new Coord(fstCoord.x - sndCoord.x, fstCoord.y - sndCoord.y);
    }
}
