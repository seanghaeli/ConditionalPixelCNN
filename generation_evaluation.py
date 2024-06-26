'''
Create all the files needed for submission on Hugging Face
'''
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import * 
from dataset import *
import os
import torch
import argparse
from pprint import pprint
# You should modify this sample function to get the generated images from your model
# This function should save the generated images to the gen_data_dir, which is fixed as 'samples'
# Begin of your code
sample_op = lambda x : sample_from_discretized_mix_logistic(x, 5)
def my_sample(model, gen_data_dir, sample_batch_size = 25, obs = (3,32,32), sample_op = sample_op):
    for label in my_bidict:
        print(f"Label: {label}")
        #generate images for each label, each label has 25 images
        sample_t = sample(model, sample_batch_size, obs, sample_op,[my_bidict[label]]*sample_batch_size)
        sample_t = rescaling_inv(sample_t)
        save_images(sample_t, os.path.join(gen_data_dir), label=label)
    pass
# End of your code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--model_name', type=str,
                        default='conditional_pixelcnn', help='Location for the dataset')
    parser.add_argument('-q', '--nr_resnet', type=int, default=2,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=100,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-o', '--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture. Higher = more flexible model')

    args = parser.parse_args()
    pprint(args.__dict__)

    ref_data_dir = "data/test"
    gen_data_dir = "samples"
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)
    #Begin of your code
    #Load your model and generate images in the gen_data_dir
    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=3, nr_logistic_mix=args.nr_logistic_mix)
    model = model.to(device)
    model_name = 'models/' + args.model_name + '.pth'
    model.load_state_dict(torch.load(model_name,map_location=torch.device(device)))
    model = model.eval()
    print('model parameters loaded')
    my_sample(model=model, gen_data_dir=gen_data_dir)
    #End of your code
    paths = [gen_data_dir, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir))
    except:
        print("Dimension {:d} fails!".format(192))
        
    print("Average fid score: {}".format(fid_score))
