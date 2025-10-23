"""
DEPRECATION NOTICE:
This file has been deprecated in favor of the unified analyzer in nba_prop_analyzer_fixed.py.
The unified analyzer now uses Expected Log Growth (ELG) scoring with dynamic fractional Kelly
sizing and prop-aware probability models.

Please use: python nba_prop_analyzer_fixed.py
"""

import sys

def main():
    print("=" * 70)
    print("⚠️  DEPRECATION WARNING")
    print("=" * 70)
    print("nba_prop_analyzer_optimized.py is deprecated.")
    print()
    print("The project has been unified into a single analyzer:")
    print("  → nba_prop_analyzer_fixed.py")
    print()
    print("New features in the unified analyzer:")
    print("  • Expected Log Growth (ELG) scoring")
    print("  • Dynamic fractional Kelly with conservative quantiles")
    print("  • Prop-aware probability models (Normal, Poisson/NegBin)")
    print("  • Top 5 per category output")
    print("  • Exposure caps for portfolio assembly")
    print("  • No artificial probability caps")
    print()
    print("Please run:")
    print("  python nba_prop_analyzer_fixed.py")
    print("=" * 70)
    
    # Optionally, delegate to the unified analyzer
    response = input("\nWould you like to run the unified analyzer now? [Y/n]: ")
    if response.lower() in ['', 'y', 'yes']:
        print("\n🔄 Launching unified analyzer...\n")
        import nba_prop_analyzer_fixed
        nba_prop_analyzer_fixed.run_analysis()
    else:
        print("\n✋ Exiting. Use 'python nba_prop_analyzer_fixed.py' when ready.")
        sys.exit(0)

if __name__ == "__main__":
    main()
