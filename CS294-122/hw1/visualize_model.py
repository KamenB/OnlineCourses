from keras.models import load_model
from keras.utils.visualize_util import plot

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
args = parser.parse_args()

model = load_model(args.model_path)
url = 'model_vizualizations/' + args.model_path.split("/")[-1].split(".")[0] + ".png"
plot(model, to_file=url, show_shapes=True)