import random
from pathlib import Path
from numpy.lib.recfunctions import structured_to_unstructured

import dv_toolkit as kit
import cocoa_flade as cocoa


class FlaDE(object):
    def __init__(self, file_path, partition, shuffle=False, denoised=True, num_samples=None):
        # load FlaDE dataset
        dataset = cocoa.FlaDE(file_path)
        elements = dataset.get_tags(key='partition', query=[f'{partition}'])

        # whether shuffle
        random.seed(42)
        if shuffle: random.shuffle(elements)

        # set number of samples
        num_samples = len(elements) if num_samples is None else num_samples

        # ready to iteration
        self.denoised = denoised
        self.path = dataset.path
        self.name = dataset.name
        self.cats = {cat['name']: cat['id'] for cat in dataset.get_cats(key='name')}
        self.elements = elements[:num_samples]
        self.dataset = dataset

    def __getitem__(self, index):
        # get item path
        element = self.elements[index]
        process = 'denoised' if self.denoised else 'raw'
        aedat_file = f"samples/{element['scene']}/{process}_clips/{element['frame']}.aedat4"

        # load aedat4 data
        reader = kit.io.MonoCameraReader(str(self.path / self.name / aedat_file))
        data = reader.loadData()
        width, height = reader.getResolution("events")

        # parse samples
        sample = {
            'events': self._parse_events_from(data['events']),
            'frames': self._parse_frames_from(data['frames']),
        }
        
        # parse targets
        targets = {
            'name': element['name'],
            'labels': [self.cats[elem.get('label')] for elem in element['boxes']],
            'bboxes': [[
                elem.get('xtl') / width,
                elem.get('ytl') / height,
                (elem.get('xbr') - elem.get('xtl')) / width,
                (elem.get('ybr') - elem.get('ytl')) / height
            ] for elem in element['boxes']],
            'resolution': (width, height)
        }

        return sample, targets

    def __len__(self):
        return len(self.elements)

    def _parse_events_from(self, events):
        # when empty
        if events.isEmpty(): return None
        
        # convert to numpy
        return structured_to_unstructured(events.numpy())

    def _parse_frames_from(self, frames):
        # when empty
        if frames.isEmpty(): return None
        
        # convert to numpy
        return frames.front().image
