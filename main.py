#!/opt/homebrew/bin/python3
"""
Name: main.py
Purpose: Transcribes MP3 files into MIDI files through the PyTorch implementation of MT3
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"

env = 'test2'

import argparse
from inference import InferenceHandler

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3 files into MIDI using MT3")
    parser.add_argument('-i', '--input', required=True, help='Path to input MP3 file')
    parser.add_argument('-o', '--output', required=True, help='Path to output MIDI file')
    args = parser.parse_args()

    input = args.input
    instruments = input.split('.')[1].split('_')[1:]
    instruments = [int(instrument) for instrument in instruments]
    print('instruments', instruments)
    num_beams = 1
    instruments_adjustment = 'AFTER'

    output = args.output

    handler = InferenceHandler('./pretrained')
    handler.inference(input, output, valid_programs=instruments, num_beams=num_beams, instruments_adjustment=instruments_adjustment)

if __name__ == "__main__":
    main()
