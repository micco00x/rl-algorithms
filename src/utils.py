import skimage
#from skimage import color, transform
import numpy as np

# Preprocess the frame as in https://github.com/pikinder/DQN/blob/master/util.py
# (captures the playing area returning a 84x84x1 8-bit color frame):
#def pong_preprocess(frame):
#    frame = frame[35:195]
#    frame = skimage.color.rgb2gray(frame) * 255
#    frame = skimage.transform.resize(frame, (84, 84))
#    return frame[:,:,np.newaxis].astype(np.uint8)

# State shape of Pong frame after preprocessing considering
# original frames have shape (210, 160, 3):
#def pong_state_shape():
#    return pong_preprocess(np.empty([210, 160, 3])).shape

def save_performances(performances_file_name, performances, description=None):
    with open(performances_file_name, "w") as performances_file:
        if description:
            performances_file.write(description + "\n")
        for p in performances:
            performances_file.write("\t".join(p) + "\n")
