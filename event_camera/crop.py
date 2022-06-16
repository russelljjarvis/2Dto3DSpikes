import argparse
import event_stream
import numpy
import pathlib

dirname = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("file_path")
parser.add_argument("left", type=int)
parser.add_argument("bottom", type=int)
parser.add_argument("width", type=int)
parser.add_argument("height", type=int)
args = parser.parse_args()
file_path = pathlib.Path(args.file_path)

left = args.left
bottom = args.bottom
assert args.left >= 0
assert args.bottom >= 0
right = args.left + args.width
top = args.bottom + args.height

decoder = event_stream.Decoder(file_path)
assert right <= decoder.width
assert top <= decoder.height
encoder = event_stream.Encoder(
    file_path.parent
    / f"{file_path.stem}_cropped_left={left}_bottom={bottom}_width={args.width}_height={args.height}.es",
    decoder.type,
    decoder.width,
    decoder.height,
)

for chunk in decoder:
    chunk = chunk[
        numpy.logical_and.reduce(
            (
                chunk["x"] >= left,
                chunk["x"] < right,
                chunk["y"] >= bottom,
                chunk["y"] < top,
            )
        )
    ]
    chunk["x"] -= left
    chunk["y"] -= bottom
    encoder.write(chunk)
