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


def save_as_png(val_array, img_file_name, channels, width, height):
    if channels == 1:
        img_array = val_array.reshape((channels, width, height))
        img_array = np.clip(img_array, 0, img_array.max())
        img_array *= 255.0 / img_array.max()
        img_array = np.rint(img_array).astype('uint8')

        try:
            os.remove(img_file_name)
        except OSError:
            pass

        im = Image.fromarray(img_array)
        im.save(img_file_name)
    else:
        val_array = val_array.ravel()[: width * height]
        img_array = val_array.reshape((width, height))
        img_array = np.clip(img_array, 0, img_array.max())
        img_array *= 255.0 / img_array.max()
        img_array = np.rint(img_array).astype('uint8')

        try:
            os.remove(img_file_name)
        except OSError:
            pass

        im = Image.fromarray(img_array)
        im.save(img_file_name)
        
    

def generate_visualization(map_file, model_file, output_path, channels, width, height):
    num_objects_to_eval = 5

    model_file_name = model_file
    encoder_output_file_name = "encoder_output_PY.txt"
    decoder_output_file_name = "decoder_output_PY.txt"
    enc_node_name = "pooling_node"
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

    # evaluate model save output
    features_si = minibatch_source['features']
    with open(os.path.join(output_path, decoder_output_file_name), 'wb') as decoder_text_file:
        with open(os.path.join(output_path, encoder_output_file_name), 'wb') as encoder_text_file:
            for i in range(0, num_objects_to_eval):
                mb = minibatch_source.next_minibatch(1)
                raw_dict = output_nodes.eval(mb[features_si])
                output_dict = {}
                for key in raw_dict.keys(): 
                    output_dict[key.name] = raw_dict[key]

                encoder_input = output_dict[input_node_name]
                encoder_output = output_dict[enc_node_name]
                decoder_output = output_dict[output_node_name]
                in_values = (encoder_input[0][0].flatten())[np.newaxis]
                enc_values = (encoder_output[0][0].flatten())[np.newaxis]
                out_values = (decoder_output[0][0].flatten())[np.newaxis]

                # write results as text and png
                np.savetxt(decoder_text_file, out_values, fmt="%.6f")
                np.savetxt(encoder_text_file, enc_values, fmt="%.6f")
                save_as_png(in_values,  os.path.join(output_path, "imageAutoEncoder_%s__input.png" % i), channels, width, height)
                save_as_png(out_values, os.path.join(output_path, "imageAutoEncoder_%s_output.png" % i), channels, width, height)

                # visualizing the encoding is only possible and meaningful with a single conv filter
                enc_dim = 7
                if(enc_values.size == enc_dim*enc_dim):
                    save_as_png(enc_values, os.path.join(output_path, "imageAutoEncoder_%s_encoding.png" % i), dim=enc_dim)

    print("Done. Wrote output to %s" % output_path)

    
if __name__=='__main__':
    this = os.path.abspath(os.path.dirname(__file__))
    folder = os.path.join(this, "101_ObjectCategories")
    
    # visualization
    map_file = os.path.join("map_file_101_ObjectCategories.txt")
    model_file = os.path.join("101_ObjectCategories", "07_Deconvolution_PY_29.model")
    output_path = "output_path"
    generate_visualization(map_file, model_file, output_path, 3, 500, 400)
