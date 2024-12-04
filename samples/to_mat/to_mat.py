"""
https://github.com/bald6354/aedat4tomat/tree/master
"""

import scipy.io as sio
import dv_toolkit as kit

class MatFile: pass

if __name__ == '__main__':
    # load data
    reader = kit.io.MonoCameraReader("/home/szd/workspace/tmp-flade/data/FlaDE/assets/F01_c02/record.aedat4")
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

    sio.savemat('./record.mat', {'aedat': aedat})   
