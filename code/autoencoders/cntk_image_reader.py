"""
An image reader on CTNK.
"""
import os
import pandas
import cntk as C
import cntk.io.transforms as xforms
from cntk.io import MinibatchSource, StreamDef, StreamDefs, ImageDeserializer
from pyquickhelper.filehelper import explore_folder_iterfile

def enumerate_map_file(folder, extension, class_mapping=None, include_unknown=False):
    """
    Detects all images in a folder, we assume the subfolder is the label.
    
    @param  folder          folder to explore.
    @param  extension       image extension
    @param  class_mapping   class mapping
    @param  include_unknown include unknwown
    @return                 enumerate label, image, num_label
    """
    memo = {}
    for name in explore_folder_iterfile(folder, pattern=".*[.]" + extension + "$"):
        if not name.endswith(extension):
            continue
        path, image = os.path.split(name)
        p, label = os.path.split(path)
        if class_mapping is None:
            new_label = label
        else:
            new_label = class_mapping.get(label, label)
        if new_label not in memo:
            memo[new_label] = len(memo)
        if class_mapping is None:
            yield label, name, memo[new_label]
        else:
            if label not in class_mapping:
                if include_unknown:
                    yield label, name, memo[new_label]
            else:
                yield label, name, memo[new_label]
                
                
def create_map_file(map_file, folder, extension, class_mapping=None, include_unknown=False):
    """
    Detects all images in a folder, we assume the subfolder is the label.
    
    @param  folder          folder to explore.
    @param  extension       image extension
    @param  class_mapping   class mapping
    @param  include_unknown include unknwown
    @return                 enumerate label, image
    """
    with open(map_file, "w") as f:
        for label, name, num_label in enumerate_map_file(folder, extension, class_mapping=class_mapping, include_unknown=include_unknown):
            f.write("{0}\t{1}\n".format(name, num_label))
            

def create_reader(map_file, channels=3, width=500, height=400, transforms=None, randomize=True):
    """
    Creates a reader on images.
    See `cntk.io.transforms <https://docs.microsoft.com/en-us/python/api/cntk.io.transforms?view=cntk-py-2.2>`_
    for available transforms.
    
    @param  map_file    file containing the image path
    @param  dest        destination folder (cache)
    @param  train       train or test
    @param  channels    number of channels
    @param  width       width (used only if transforms is None)
    @param  height      height (used only if transforms is None)
    @param  transforms  transforms or default one if None
    @param  randomize   randomize
    """
    df = pandas.read_csv(map_file, sep="\t", header=None)
    df.columns = [["image", "label"]]
    num_classes = len(set(df.label))
    if transforms is None:
        transforms = [xforms.scale(width=width, height=height, 
                        channels=channels, interpolations='linear')]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
            features=C.io.StreamDef(field='image', transforms=transforms),
            labels=C.io.StreamDef(field='label', shape=num_classes))),
            randomize=randomize)
    