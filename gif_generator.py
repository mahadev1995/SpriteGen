import imageio
import glob

paths = glob.glob('./plots/*.png')
paths.sort()

images = []
for filename in paths:
    images.append(imageio.imread(filename))
imageio.mimsave('training.gif', images, duration=600)