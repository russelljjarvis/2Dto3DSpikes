import argparse
import event_stream
import matplotlib
import matplotlib.pyplot
import numpy
import pathlib

matplotlib.use("agg")

parser = argparse.ArgumentParser()
parser.add_argument("file_path")
args = parser.parse_args()
file_path = pathlib.Path(args.file_path)

decoder = event_stream.Decoder(file_path)
events = numpy.concatenate([chunk for chunk in decoder])

matplotlib.pyplot.rc("font", size=18, family="Times New Roman")
figure = matplotlib.pyplot.figure(figsize=(16, 9), dpi=80)
subplot = figure.add_subplot(111)
subplot.set_xlabel("Time (s)")
subplot.set_ylabel("Pixel address")
subplot.spines["top"].set_visible(False)
subplot.spines["right"].set_visible(False)
subplot.scatter(
    x=events["t"] / 1e6,
    y=events["x"] + events["y"] * decoder.width,
    s=0.5,
    marker=".",
    c="#000000",
)
figure.savefig(str(file_path.parent / f"{file_path.stem}_raster_plot.png"))
matplotlib.pyplot.close(figure)
