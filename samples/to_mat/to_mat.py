"""
https://github.com/bald6354/aedat4tomat/tree/master
"""

import argparse
import scipy.io as sio
import dv_toolkit as kit

class MatFile: 
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Convert .aedat4 to .mat")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input .aedat4 file path"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output .mat file path"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load data
    reader = kit.io.MonoCameraReader(args.input)
    data, resolution = reader.loadData(), reader.getResolution("events")

    # meta
    aedat = MatFile()

    # resolution
    aedat.resolution = resolution

    # events
    aedat.events = MatFile()
    aedat.events.timestamp = data['events'].timestamps().tolist()
    aedat.events.x = data['events'].xs().tolist()
    aedat.events.y = data['events'].ys().tolist()
    aedat.events.polarity = data['events'].polarities().tolist()

    # frames
    aedat.frames = MatFile()
    aedat.frames.timestamp = []
    aedat.frames.image = []
    
    for frame in data['frames']:
        aedat.frames.timestamp.append(frame.timestamp)
        aedat.frames.image.append(frame.image)

    # imus
    aedat.imus = MatFile()
    
    # triggers
    aedat.triggers = MatFile()

    sio.savemat(args.output, {'aedat': aedat})
