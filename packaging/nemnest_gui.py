"""Entry point of the standalone Windows program (built with PyInstaller)."""
import multiprocessing
import sys

from nemnest import gui

if __name__ == "__main__":
    multiprocessing.freeze_support()     # the STL readers run in worker processes
    sys.exit(gui.main(sys.argv[1:]))
