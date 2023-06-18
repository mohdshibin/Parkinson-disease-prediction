def extractAudio(audio):
    pass

#Measure pitch of all wav files in directory
import glob
import numpy as np
import pandas as pd
import parselmouth

from parselmouth.praat import call
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# import module
import pickle as pkl
#import sklearn
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import skimage
from skimage import feature
import cv2  




def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    maxf0 = call(pitch, "Get maximum", 0, 0, unit, "Parabolic") # get max pitch
    minf0 = call(pitch, "Get minimum", 0, 0, unit, "Parabolic") # get min pitch
    # freq_range = pitch.get_range()
    # maxf0 = freq_range[1]
    # minf0 = freq_range[0]
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    nhr = 1/hnr
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    res = list(map(lambda x: "{:.3f}".format(float(x)), [meanF0, maxf0, minf0, stdevF0, hnr, nhr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]))
    res[0] = "{:.3f}".format(float(res[0]))
    res[1] = "{:.3f}".format(float(res[1]))
    res[2] = "{:.3f}".format(float(res[2]))
    res[4] = "{:.3f}".format(float(res[4]))
    res[11] = "{:.3f}".format(float(res[11]))
    print(res)
    

    # return meanF0, maxf0, minf0, stdevF0, hnr, nhr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer
    return res




# def runPCA(df):
#     #Z-score the Jitter and Shimmer measurements
#     features = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
#                 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
#     # Separating out the features
#     x = df.loc[:, features].values
#     # Separating out the target
#     #y = df.loc[:,['target']].values
#     # Standardizing the features
#     x = StandardScaler().fit_transform(x)

    
#     #PCA
#     pca = PCA(n_components=2)
#     principalComponents = pca.fit_transform(x)
#     principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
#     principalDf
#     return principalDf



def quantify_image(image):
  features = feature.hog(image,orientations=9,
                pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True,block_norm="L1")

  return features



