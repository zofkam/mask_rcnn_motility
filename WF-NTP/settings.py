### Input
#filename = "C:/Data/wf_ntp/WF-NTPv2.0-master/output/intelj_generated_005.AVI"
filename = "C:/Users/Administrator/Downloads/25z004.avi"
start_frame = 0
limit_images_to = 200
fps = 20.0
px_to_mm = 0.00163
darkfield = False
stop_after_example_output = False

### Output
save_as = "C:/Data/wf_ntp/WF-NTPv2.0-master/output/"
output_overlayed_images = 0
font_size =  8
fig_size = (20,20)
scale_bar_size = 1.0
scale_bar_thickness = 7

### Z-filtering
use_images = 100
use_around = 5
Z_skip_images = 1

### Thresholding
keep_dead_method = True # False
std_px = 128 #64
threshold = 7 #12
opening = 3 #1
closing = 3 #3
skeletonize = False
prune_size = 0
do_full_prune = False

### Locating
# slightly reduced min value from annotated (2500), cause we can have transparent worms
min_size = 2000 #500 #1000
max_size = 7800 #14000 #8000
minimum_ecc = 0.8 #0.93

### Form trajectories
max_dist_move = 100
min_track_length = 5
memory = 5

### Bending statistics
bend_threshold = 2.0
minimum_bends = 0.0

### Velocities
frames_to_estimate_velocity = 3

### Dead worm statistics
maximum_bpm = 0.5
maximum_velocity = 0.1

### Regions
regions = {}
