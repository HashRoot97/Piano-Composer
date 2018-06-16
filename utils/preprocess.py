from music21 import converter, instrument, note, chord
import glob
import numpy as np
import keras
import np_utils
import os

notes = []
print('Generating Notes ...')
for file in glob.glob("./../midi_songs/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = None

    parts = instrument.partitionByInstrument(midi)

    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print('Notes Generated')
print('Generating Dataset ... ')

pitch_names = sorted(set(item for item in notes))
n_vocab = len(pitch_names)

note_to_int = dict(((number, note) for note, number in enumerate(pitch_names)))
sequence_length = 100
network_input = []
network_output = []
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

network_input = np.array(network_input)
network_output = np.array(network_output)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_output = keras.utils.to_categorical(network_output)
print(network_input.shape)
print(network_output.shape)
os.system('mkdir ./../NumpyDataset')
np.save('./../NumpyDataset/Input-Tensor.npy', network_input)
np.save('./../NumpyDataset/Output-Tensor.npy', network_output)

print('Numpy Dataset Generated')