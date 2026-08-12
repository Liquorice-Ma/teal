#!/bin/bash
# kill teal training processes without matching this script itself
for p in $(pgrep -f "bin/python teal.py"); do kill $p 2>/dev/null; done
for p in $(pgrep -f verify_noise); do kill $p 2>/dev/null; done
for p in $(pgrep -f eval_denoised); do kill $p 2>/dev/null; done
sleep 2
echo remaining: $(pgrep -f "bin/python teal.py" | wc -l)
