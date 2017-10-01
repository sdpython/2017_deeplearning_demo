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
            

