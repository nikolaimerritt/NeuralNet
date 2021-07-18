using System;
using System.Collections.Generic;
using System.Text;

namespace CatchTheCheeseGame
{
    public interface IWasteman
    {

    }

    public class Nikolai : IWasteman { public void poo() { } }
    public class DizzyNikolai : Nikolai { }

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

        public static void Test()
        {
            IWasteman waste = new DizzyNikolai();
            switch (waste)
            {
                case DizzyNikolai dizzy: break;
                case Nikolai nikolai: break;
                default: break;
            }

            Dictionary<int, string> poo = new();
            if (poo.TryGetValue(5, out string res))
            {

            }
        }
    }
}
