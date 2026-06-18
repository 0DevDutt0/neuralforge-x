# VHS tape — record a NeuralForge-X quickstart GIF.
#
# Reproducible terminal recording (no manual capture). Install VHS
# (https://github.com/charmbracelet/vhs), then from the repo root run:
#
#     vhs scripts/demo.tape
#
# Output lands in docs/assets/gifs/ and is embedded in the README.

Output docs/assets/gifs/quickstart.gif

Set Shell "bash"
Set FontSize 18
Set Width 1100
Set Height 640
Set Theme "Dracula"
Set Padding 18

Type "python examples/01_quickstart.py"
Sleep 600ms
Enter
Sleep 5s

Type "python examples/02_numpy_vs_rust_benchmark.py"
Sleep 600ms
Enter
Sleep 8s
