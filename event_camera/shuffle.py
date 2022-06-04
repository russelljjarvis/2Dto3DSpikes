import argparse
import event_stream
import json
import numpy
import pathlib

dirname = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("file_path")
args = parser.parse_args()
file_path = pathlib.Path(args.file_path)

decoder = event_stream.Decoder(file_path)
encoder = event_stream.Encoder(
    file_path.parent / f"{file_path.stem}_shuffled.es",
    decoder.type,
    decoder.width,
    decoder.height,
)

x_map = numpy.arange(0, decoder.width, dtype="<u2")
y_map = numpy.arange(0, decoder.height, dtype="<u2")
numpy.random.shuffle(x_map)
numpy.random.shuffle(y_map)
with open(
    file_path.parent / f"{file_path.stem}_shuffled_x_map.json", "w"
) as x_map_output:
    json.dump(x_map.tolist(), x_map_output)
with open(
    file_path.parent / f"{file_path.stem}_shuffled_y_map.json", "w"
) as y_map_output:
    json.dump(y_map.tolist(), y_map_output)
for chunk in decoder:
    chunk["x"] = x_map[chunk["x"]]
    chunk["y"] = y_map[chunk["y"]]
    encoder.write(chunk)
