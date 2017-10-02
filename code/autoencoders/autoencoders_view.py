"""
Train an autoencoders with `CNTK <https://github.com/Microsoft/CNTK>`_.
"""
import pandas
import numpy as np
import sys
import os
import cntk as C
from cntk import load_model
from cntk_image_reader import create_reader
from cntk.ops import combine
from PIL import Image
from autoencoders_train_pink import PinkActivation


def save_as_png(output_path, i, *arrays):
    file_names = []
    for sub, val_array in enumerate(arrays):
        if len(val_array.shape) != 3:
            raise ValueError("Shape dimension should be 3 not '{0}'".format(val_array.shape))
        img_file_name = os.path.join(output_path, "img_{0}_l{1}.png".format(i, sub))
        file_names.append((img_file_name, sub))
        try:
            os.remove(img_file_name)
        except OSError:
            pass

        height,width = val_array.shape[1:]
        
        if val_array.shape[0] == 1:
            img_array = val_array * 255.0 / val_array.max()
            img_array = np.rint(img_array).astype('uint8')

            im = Image.fromarray(img_array)
            im.save(img_file_name)
        else:
            channels = val_array.shape[0]
            rgbArray = np.zeros((height,width, channels), 'uint8')
            maxi = val_array.max()
            for ic in range(channels):
                rgbArray[:,:, ic] = val_array[2-ic,:,:] * 255.0 / maxi
            img = Image.fromarray(rgbArray)
            img.save(img_file_name)
            
            maxi = [val_array[2-ic,:,:].max() for ic in  range(channels)]
            if min(maxi) < max(maxi) / 3:
                print(maxi)
                for ic in range(channels):
                    rgbArray[:,:, ic] = val_array[2-ic,:,:] * 255.0 / maxi[ic]
                img = Image.fromarray(rgbArray)
                img_file_name = os.path.join(output_path, "img_{0}_l{1}_eq.png".format(i, sub))            
                img.save(img_file_name)
                file_names.append((img_file_name, "{0}r".format(sub)))
            
    return file_names
        
    

def generate_visualization(map_file, model_file, output_path, 
                                            channels, width, height, suffix,
                                            num_objects_to_eval=5, skip=0,
                                            save=None):
    model_file_name = model_file
    encoder_output_file_name = "encoder_output_PY.txt"
    decoder_output_file_name = "decoder_output_PY.txt"
    enc_node_name = "ae_node"
    input_node_name = "input_node"
    output_node_name = "output_node"
    
    minibatch_source = create_reader(map_file, channels=channels, width=width, 
                                 height=height, randomize=False)
    
    # load model and pick desired nodes as output
    loaded_model = load_model(model_file)
    output_nodes = combine(
        [loaded_model.find_by_name(input_node_name).owner,
         loaded_model.find_by_name(enc_node_name).owner,
         loaded_model.find_by_name(output_node_name).owner])
         
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    df = pandas.read_csv(map_file, sep="\t", header=None)
    df.columns = [["image", "label"]]
                
    # open HTML file
    if save is not None:
        save.write("<h1>{0}</h1>\n".format(suffix))

    # evaluate model save output
    features_si = minibatch_source['features']
    for i in range(0, num_objects_to_eval + skip):
        mb = minibatch_source.next_minibatch(1)
        if i < skip:
            continue
        f.write("<hr><h2>Image {0}</h2>\n".format(i))
        raw_dict = output_nodes.eval(mb[features_si])
        output_dict = {}
        for key in raw_dict.keys(): 
            output_dict[key.name] = raw_dict[key]

        encoder_input = output_dict[input_node_name]
        encoder_output = output_dict[enc_node_name]
        decoder_output = output_dict[output_node_name]
        
        in_values = encoder_input[0]
        enc_values = encoder_output[0]
        out_values = decoder_output[0]

        # write results as text and png
        files = save_as_png(output_path, i, in_values, enc_values, out_values)
        orig = df.loc[i, "image"]
        orig = os.path.relpath(orig, os.path.abspath(os.path.dirname(__file__)))
        print(orig)
        if not os.path.exists(orig):
            raise FileNotFoundError(orig)
        f.write('<img src="{0}" />\n'.format(orig))
        for name, legend in files:
            f.write('{1}<img src="{0}" alt="{1}"/>\n'.format(name, legend))

    print("Done. Wrote output to %s" % output_path)

    
if __name__=='__main__':
    this = os.path.abspath(os.path.dirname(__file__))
    folder = os.path.join(this, "101_ObjectCategories")
    
    # visualization
    map_file = os.path.join("map_file_101_ObjectCategories.txt")
    output_path = "output_path"
    channels = 3
    suffixes = [_ for _ in os.listdir("models") if "x" in _ and ".25" not in _]
    with open("report.html", "w") as f:
        f.write("<html><body>\n")
        for suffix in suffixes:
            if "pink" in suffix:
                continue
            print("------------", suffix)
            model_file = os.path.join("models", suffix, "ae_99.model")
            width, height = [int(_) for _ in suffix.split("_")[1].split("x")]    
            generate_visualization(map_file, model_file, output_path, channels, width, height, 
                                             suffix, skip=220, save=f)
        f.write("</body></html>\n")
