# We have 6 input modes, each requiring a different way to handle it and
# load it into the loader (and also data augmentation)
MODALITIES = ["Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton"]
NUM_CLASSES = 40


# Number of sampled frames for the vision (i.e. depth maps, thermal imaging, infrared)
# we resample as the clips are short, and images are nearly the same in a small temporal
# window
VISION_FRAMES = 16
# The re-shaping size for the images ^ as above
VISION_SIZE = 112

SKELETON_FRAMES = 64
NUM_JOINTS = 17
# this is for augmentation: Since we are training backbones individually, we can train the
# skeleton backbone individually too, so we do not have to unify augmented data across channels.
# thus we can rotate the skeleton along the y (height) axis freely, to represent "rotated"
# images.
SKELETON_UP_AXIS = 2

IMU_LEN = 400
IMU_DEVICES = ["WTC", "WTLA", "WTRA", "WTLL", "WTRL"]
IMU_CH = 6 * len(IMU_DEVICES)

RADAR_FEATURES = 12

GYRO_UNIT_DPS = 250.0
SNR_UNIT = 100.0
NOISE_UNIT = 100.0

# Per channel mean and std fitted on the training split with fit_stats.
# Applied to IMU and Radar, whose channels mix heterogeneous physical units.
# Vision stays in [0, 1]: the first BatchNorm absorbs fixed input affines for
# networks trained from scratch.
STATS_PATH = "channel_stats.json"
# ^ This will be created in the cache directory
_STATS = None

# Augmentation strengths. Conventional starting magnitudes, tune against the
# user split validation.
AUG_CROP_FACTOR = 1.15
AUG_GAIN_RANGE = (0.8, 1.2)
AUG_SKEL_ROT_RAD = 3.14159265 / 6
AUG_SKEL_NOISE = 0.01
AUG_IMU_GAIN_RANGE = (0.9, 1.1)
AUG_IMU_NOISE = 0.02
AUG_RADAR_NOISE = 0.02

RADAR_LEN = 32           # temporal bins (same role as before)
RADAR_MAX_PTS = 8       # max points kept per bin, pad/truncate to this
RADAR_POINT_FEATS = 6    # x, y, z, v, snr_norm, noise_norm
AUG_POINT_JITTER = 0.02  # meters, gaussian jitter on xyz (train only)
AUG_POINT_DROPOUT = 0.15 # probability of zeroing out a point (train only)


# Data input shapes
SHAPES = {
    "Depth_Color": (VISION_FRAMES, 3, VISION_SIZE, VISION_SIZE),
    "IR": (VISION_FRAMES, 1, VISION_SIZE, VISION_SIZE),
    "Thermal": (VISION_FRAMES, 1, VISION_SIZE, VISION_SIZE),
    "IMU": (IMU_CH, IMU_LEN),
    "Radar": (RADAR_FEATURES, RADAR_LEN),
    "Skeleton": (SKELETON_FRAMES, NUM_JOINTS, 3),
}
