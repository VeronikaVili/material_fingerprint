# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from torch import nn
import torch
import numpy as np
import cv2
from skimage.color import rgb2lab, lab2lch, lab2rgb, lch2lab
import scipy as sp
import math
from sklearn.preprocessing import minmax_scale
from findpeaks import findpeaks
from matplotlib import pyplot as plt
import copy

# File paths
# Current version of model parameters
MODEL_PARAM_1 = './modelParams'
# Current version of STDs and MEANs used in standardization
file = './statsMeanStd.txt'
MEANS = np.loadtxt(file, dtype=float, usecols=0, delimiter=" ")
STDS = np.loadtxt(file, dtype=float, usecols=1, delimiter=" ")


# Constants
# Computational stats names
STAT_NAMES = ['Max','Min','Mean', 'Variance', 'Skewness', 'Kurtosis', 'Directionality',
              'Low frequencies', 'Middle frequencies', 'High frequencies',
              'Mean chroma', 'Pattern strength', 'Pattern number', 'Colors number']
# Rating stats names
RATING_NAMES = ['Color vibrancy', 'Surface roughness','Pattern complexity', 'Striped pattern',
                'Checkered pattern', 'Brightness', 'Shininess', 'Sparkle', 'Hardness',
                'Movement effect', 'Scale of pattern', 'Naturalness', 'Thickness',
                'Multicolored', 'Value', 'Warmth']
# Rearranging for polar plot
RATING_CHANGE = [6, 7, 2, 3, 4, 1, 10, 13, 0, 5, 11, 14, 15, 12, 8, 9]
# Parameter for directionality computation
CIRCLES = 32
SECTORS = 24
# Parameter for PSD frequency
BORDERS = [0, 8, 64, 256]
# Parameter for size of image
# Should cooperate with Fourier transform - best is powers of 2
FOURIER_SIZE = 512

# ------------------------------------------------------------------
# Functions for reading and modifying image
# ------------------------------------------------------------------

# image - image to crop
# N - size of both sides of the new image (it will be a square)
# returns cropped image
def cut_image_middle_N(image, N):
    # Checking conditions for N
    if N > image.shape[0] or N > image.shape[1] or N < 0:
        print("Error: N is wrong.")
        return None
    
    # Calculating the middle
    middle = (int(math.floor(image.shape[0]/2)), int(math.floor(image.shape[1]/2)))
    N_half = math.floor(N/2)
    
    # Cutting the image in 2 or 3 dimensions
    if len(image.shape) == 2:
        image_cut = image[middle[0]-N_half:middle[0]+N_half, middle[1]-N_half:middle[1]+N_half]
    elif len(image.shape) == 3:
        image_cut = image[middle[0]-N_half:middle[0]+N_half, middle[1]-N_half:middle[1]+N_half, :]
    else:
        print("Error: Image dimension needs to be 2 or 3")
        return None
    
    return image_cut

# path - path to image location
def open_vid_image(path):
    # Reading image
    image = cv2.imread(path)
    # Converting BGR input to RGB image
    image = image[...,::-1]
    return image

# image - image to transform
# from_space - color space to convert from - works for RGB, LAB, LCH
# to_space - color space to convert to - works for RGB, LAB, LCH
def transfer_color_space(image, from_space, to_space):
    if from_space.lower() == "rgb":
        if to_space.lower() == "lch":
            image = rgb2lab(image, "D65")
            image = lab2lch(image)
            return image
        elif to_space.lower() == "lab":
            image = rgb2lab(image, "D65")
            return image
        else:
            print("Error: To space LAB or LCH")
            return None
    elif from_space.lower() == "lab":
        if to_space.lower() == "lch":
            image = lab2lch(image)
            return image
        elif to_space.lower() == "rgb":
            image = lab2rgb(image)
            return image
        else:
            print("Error: To space RGB or LCH")
            return None
    elif from_space.lower() == "lch":
        if to_space.lower() == "lab":
            image = lch2lab(image)
            return image
        elif to_space.lower() == "rgb":
            image = lch2lab(image)
            image = lab2rgb(image)
            return image
        else:
            print("Error: To space RGB or LAB")
            return None
    else:
        print("Error: From space RGB, LAB or LCH")
        return None
        
    
# ------------------------------------------------------------------    
# Statistic computation
# ------------------------------------------------------------------

# image - image to preprocess
# returns square image size SIZE with one channel
def prepare_image_for_fourier(image, SIZE):
    image_cut = cut_image_middle_N(image, SIZE)
    # Fourier transform requires only one channel
    if len(image.shape) > 2 :
        image_cut = image_cut[:, :, 0]
    return image_cut
    
# image - image to count PSD of, needs to be a square, best is power of 2
# returns the binned frequency values, binned amplitude values
def count_only_psd(image):
    # Size of image
    npix = image.shape[0]
    # Fourier transform
    fourier_image = np.fft.fftn(image)
    # Getting the amplitudes as one number from complex numbers
    fourier_amplitudes = np.abs(fourier_image)**2
    # Getting corresponding frequencies
    kfreq = np.fft.fftfreq(npix) * npix
    kffreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kffreq2D[0]**2 + kffreq2D[1]**2)
    # Flattening the data to 1D array
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    # Setting up bins
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    # Binning the values of amplitudes
    Abins, _, _ = sp.stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
    # Muliplying by space of the bin
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins
    
# image - image to count mean frequency of
# borders - list of borders between which the mean of the frequency will be computed
# returns a list of means of amplitudes of frequencies between borders
def count_freq(image, borders):
    mean_all = []
    # Cutting image around the middle
    # Needs to be square and best is a power of 2
    image_cut = prepare_image_for_fourier(image, FOURIER_SIZE)
    # Getting the PSD
    kvals, Abins = count_only_psd(image_cut)
    # Iterating through the area set by borders
    for i in range(len(borders)-1):
        # Calculating the mean of the amplitudes between borders
        mean_all.append(np.mean(Abins[borders[i]:borders[i+1]]))
    return mean_all
    
# image - image from which to get amplitude spectrum by Fourier transform, only 1 channel and square size
# returns values of amplitudes in the size of the original image
def count_fourier_image(image):
    # Image must be a square size
    if image.shape[0] != image.shape[1]:
        print("Image must have same width and height")
        return None
    # Size of image
    npix = image.shape[0]
    # Fourier transform
    fourier_image = np.fft.fftn(image)
    # Getting the amplitudes as one number from complex numbers
    fourier_amplitudes = np.abs(fourier_image)**2
    # Shifting amplitude spectrum so low frequencies are in the middle
    # Getting the log of the data for better plotting
    return np.fft.fftshift(fourier_amplitudes)
    
# shape - shape of the original image to mask
# centre - tuple of centre coordinates of the circle to mask (middle of the image)
# radius - size of the circle to mask
# angle_range - tuple of range of the circle
# returns mask of the specified parameters
def sector_mask(shape,centre,radius,angle_range):
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)
    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)
    # circular mask
    circmask = r2 <= radius*radius
    # angular mask
    anglemask = theta <= (tmax-tmin)
    return circmask*anglemask
    
# image - image to compute, must have only 1 channel and be prepared for Fourier transform size-wise
# sectors_num - number of circular sections to divide the image into, computed only on half of them
# cir_num - number of circles to divide the image into
# start_num - starting angle for the sectors - set to 270 to start at the top
# returns means of given areas and the size of the areas
def mean_circular_sectors_cut(image_cut, sectors_num, cir_num, start_num=270):
    # Setting up the containers, only half of the image is used - sectors are divided by 2
    mean_all = np.zeros((cir_num, math.ceil(sectors_num/2)))
    sum_all = np.zeros((cir_num, math.ceil(sectors_num/2)))
    
    # Size of 1 sector in angles
    sector_size = int(360/sectors_num)
    
    # Iterate through half of the sectors
    for i in range(math.ceil(sectors_num/2)):
        # Iterate through dividing circles
        for j in range(cir_num):
            # Compute the amplitude spectrum
            fourier = count_fourier_image(image_cut)
            
            # Find the middle of the image
            middle = (int(math.floor(fourier.shape[0]/2)), int(math.floor(fourier.shape[1]/2)))
            
            # Getting the big mask that the smaller will be subtracted from to create areas between two circles
            # The size of the circle is size of the image divided by the number of the circles
            if j != (cir_num-1):
                big_size = math.floor(middle[0]/cir_num)*(j+1)
            # The last circle is the whole image - circle is the size of the image
            else:
                big_size = middle[0]
            big_mask = sector_mask(fourier.shape, (middle[0], middle[1]), big_size, (start_num+i*sector_size, (start_num+sector_size)+i*sector_size))
            
            # Getting the small mask that will be subtracted from the bigger to create areas between two circles
            small_size = math.floor(middle[0]/cir_num)*(j)
            small_mask = sector_mask(fourier.shape, (middle[0], middle[1]), small_size, (start_num+i*sector_size, (start_num+sector_size)+i*sector_size))
            
            # Masking the image - area of the big mask is the only left, then the small mask is subtracted
            # Big mask - in
            fourier[~big_mask] = 0
            # Small mask - out
            fourier[small_mask] = 0
            
            # Mean of the selected area
            mean_mask = fourier > 0
            mean_all[j,i] = np.mean(fourier[mean_mask])
            
            # Size of the area
            sum_all[j,i] = np.sum(mean_mask, axis=(0, 1))
            
    return mean_all, sum_all
    
# mean_all_mult - directionality matrix (multiplied means and sums of Fourier sectors from mean_circular_sectors_cut)
def count_directionality_cut(mean_all_mult):
    # Number of columns - of sections
    n = mean_all_mult.shape[1]
    # Get sum of the columns
    all_sum = np.sum(mean_all_mult, axis=0)
    # Select the maximum column values
    max_col = np.max(all_sum)
    # Count the directionality
    direct = np.sum(max_col - all_sum)/(n * max_col)

    return direct

# mean_mult - directionality matrix (multiplied means and sums of Fourier sectors from mean_circular_sectors_cut)   
def count_pattern_str(means_mult):
    # Compute mean of columns - info about the whole sector of Fourier from center to edge - through all frequencies
    means = np.mean(means_mult, axis=0)
    # Find the maximum
    maxim = np.argmax(means)
    ratio = []
    for i in range(len(means)):
        if i != maxim:
            # Compute the ratio between the sector and the max value sector
            ratio.append(means[maxim]/means[i])
    # Mean of the ratios - how on average is the value different from the max
    return np.mean(ratio)
    
# image - image to count statistics about pattern for
def count_pattern(image):   
    # Prepare image for Fourier transform (cut and select first channel)
    image_cut = prepare_image_for_fourier(image, FOURIER_SIZE)
    # Compute the Fourier image sectors - their mean values and sizes
    means, sums = mean_circular_sectors_cut(image_cut, SECTORS, CIRCLES)
    # Multiply the means by the size of the sector
    means_mult = np.multiply(means, sums)
    
    # Directionality computation
    direct = count_directionality_cut(means_mult)
    
    # Pattern stregth computation
    patt_str = count_pattern_str(means_mult)
    
    # Pattern number computation
    # Compute sum of columns - info about the whole sector of Fourier from center to edge - through all frequencies
    all_sum = np.sum(means_mult, axis=0)
    # Scaling to (0,1) - just need information about peaks not their size
    all_sum_red = minmax_scale(all_sum)
    # Find peaks
    fp = findpeaks(method="topology", verbose=0)
    local_max = fp.fit(all_sum_red)
    # Peaks values
    peaks = local_max["persistence"][["y", "score"]].values
    # Select only the important peaks - higher score than 0.5
    peaks_new = [i[0] for i in peaks if i[1] > 0.5]
    # If both peaks at the ends are found, the end one is deleted - ensures the "circle" nature, connect them into one peak
    if 0 in peaks_new and 11 in peaks_new:
        peaks_new.remove(11)
        
    return direct, len(peaks_new), patt_str

# image - image to count multicolor statistics for    
def count_multicolored(image):
    # 1 - A color channel
    # 2 - B color channel
    src_X = image[:, :, 1].flatten()
    src_Y = image[:, :, 2].flatten()
    
    # Calculating color space data as a 2D histogram
    Z, xedges, yedges = np.histogram2d(src_X, src_Y, bins=[np.linspace(-128, 128, 129), np.linspace(-128, 128, 129)], weights=image[:, :, 0].flatten())
    
    # Finding peaks in the 2D histogram
    # Important parameter "limit", to limit the score of peaks - suppresses noise
    fp = findpeaks(limit=30, verbose=0)
    local_max = fp.fit(Z)
    
    # Return number of peaks for image
    return (len(local_max["persistence"][["x", "y"]].values))
    
# image - image to count all statistics for
def all_stats_one_image(image):
    # Read image into LCH color space
    image = transfer_color_space(image, "rgb", "lch")

    COL=0
    COL_CHR=1
        
    # Maximum - 99 percentile
    max_val = np.percentile(image[:, :, COL], 99, axis=(0, 1))
    # Minimum - 1 percentile
    min_val = np.percentile(image[:, :, COL], 1, axis=(0, 1))
    # Mean
    mean_val = np.mean(image[:, :, COL], axis=(0, 1))
    # Variance
    var_val = np.var(image[:, :, COL], axis=(0, 1))
    # Skewness
    ske_val = sp.stats.skew(image[:, :, COL], axis=(0, 1))
    # Kurtosis
    kur_val = sp.stats.kurtosis(image[:, :, COL], axis=(0, 1))
    # Frequency analysis by PSD spectrum - BORDERS separate low, middle, high
    freq_tmp = count_freq(image[:, :, COL], BORDERS)
    freq_low = freq_tmp[0]
    freq_mid = freq_tmp[1]
    freq_high = freq_tmp[2]
    # Pattern information computation
    tmp = count_pattern(image[:, :, COL])
    # Directionality
    dir_val = tmp[0]
    # Pattern number
    pattern_num_val = tmp[1]
    # Pattern strength
    pattern_str_val = tmp[2]
    # Mean chroma - weighted by luminance
    mean_val_chr = np.mean((image[:, :, COL] * image[:, :, COL_CHR]), axis=(0, 1))
        
    # Read image into LAB color space
    image = transfer_color_space(image, "lch", "lab")
    # Number of colors
    multicolor_val = count_multicolored(image)
    
    return np.array([max_val, min_val, mean_val, var_val, ske_val, kur_val, dir_val, 
                    freq_low, freq_mid, freq_high, 
                    mean_val_chr, pattern_str_val, pattern_num_val, multicolor_val])
                    
# ------------------------------------------------------------------
# Rating statistics prediction
# ------------------------------------------------------------------
# Module definition
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
    )
    def forward(self, x):
        return self.layers(x)
        
# Module preparation
input_dim = 28
hidden_dim = 16
output_dim = 16
MODEL = MLP(input_dim, hidden_dim, output_dim)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL.to(device)

MODEL.load_state_dict(torch.load(MODEL_PARAM_1))

def prepare_image_photo(image):
    image = cv2.resize(image, dsize=(FOURIER_SIZE, FOURIER_SIZE), interpolation=cv2.INTER_NEAREST)
    return image

# Predict ratings for general images - like photos taken with a mobile phone
# It is expected that area with the same size as training data is already selected
# Images should be in the RGB space
# image_1 - NON-SPECULAR image, light from the side
# image_2 - SPECULAR image, light from the front
def predict_ratings_changed_photo(image_1, image_2):
    res = dict()
    # Resizing images to appropriate size
    # image_1 = cv2.resize(image_1, dsize=(FOURIER_SIZE, FOURIER_SIZE), interpolation=cv2.INTER_NEAREST)
    # image_2 = cv2.resize(image_2, dsize=(FOURIER_SIZE, FOURIER_SIZE), interpolation=cv2.INTER_NEAREST)
    image_1 = prepare_image_photo(image_1)
    image_2 = prepare_image_photo(image_2)

    # Count stats
    stats_1 = all_stats_one_image(image_1)
    stats_2 = all_stats_one_image(image_2)
    
    # Concatenate to 28 length
    stats = np.concatenate((stats_1, stats_2))
    res["ORIGINAL_STATS"] = stats
    # Normalize
    stats = (stats-MEANS)/STDS
    res["NORMALIZED_STATS"] = stats
    
    # Make stats array corresponding input for given model
    in_batch = torch.tensor(stats, dtype=torch.float32)
    # Send to device
    in_batch = in_batch.to(device)
    # Get output
    with torch.no_grad():
        output = MODEL(in_batch)

    res["PREDICT_RATINGS"] = output.cpu().numpy()
    return res
    
# ------------------------------------------------------------------
# Printing results
# ------------------------------------------------------------------
# Plotting results as a line graph
def plot_res(data, colors=["blue"], labels=["Data"], SIZE=[15, 8], YLIM=[-2.5, 2.5, 0.5]):
    plt.figure(figsize=(SIZE[0], SIZE[1]))
    for i in range(len(data)):
        plt.plot(RATING_NAMES, data[i], color=colors[i], marker="o", label=labels[i])
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xticks(rotation=90)
    plt.ylim(YLIM[0], YLIM[1])
    plt.yticks(np.arange(YLIM[0], YLIM[1]+YLIM[2], YLIM[2]))
    plt.show()

# General function for showing images
def show_images(images, SIZE=[20, 3]):
    fig, ax = plt.subplots(1, len(images), figsize=(SIZE[0], SIZE[1]))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_axis_off()
    plt.show()

# Polar plot
def polar_plot(data_all_old, COLORS=["blue", "red", "green"], RATINGS=RATING_NAMES, LABELS=None, 
                    title=None, order=None, save=None, noticks=False,
                    SIZE=[15, 10], ZERO_WIDTH=0.01, ZERO_COLOR="black", LINE_WIDTH=2,
                    LABEL_SIZE_X=15, LABEL_SIZE_Y=15, YLIM=[-2.5, 2.5], LEGEND_SIZE=15,
                    TITLE_SIZE=15, FILL_RATE=0.05):
    # Rearranging order
    data_all = copy.deepcopy(data_all_old)
    if order is not None:
        for i in range(len(data_all)):
            data_all[i] = data_all[i][order]
        RATINGS = np.array(RATINGS)[order]
    
    fig = plt.figure(figsize=(SIZE[0], SIZE[1]))
    ax = fig.add_subplot(111, polar=True)
    theta = np.linspace(0, 2 * np.pi, len(RATINGS), endpoint=False)
    theta = np.concatenate((theta, [theta[0]]))
    
    # Black circle around 0
    ax.fill_between(np.linspace(0, 2*np.pi, 100), -ZERO_WIDTH, ZERO_WIDTH, color=ZERO_COLOR, zorder=10)
    
    # Data plotting
    for idx, data in enumerate(data_all):
        data = np.concatenate((data, [data[0]]))
        if LABELS is not None:
            ax.plot(theta, data, marker="o", color=COLORS[idx], label=LABELS[idx], linewidth=LINE_WIDTH)
        else:
            ax.plot(theta, data, marker="o", color=COLORS[idx], linewidth=LINE_WIDTH)
        ax.fill(theta, data, color=COLORS[idx], alpha=FILL_RATE)

    theta = np.linspace(0, 2 * np.pi, len(RATINGS), endpoint=False)
    ax.set_thetagrids((theta * 180/np.pi), RATINGS)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(direction='clockwise')
    
    # Positioning the labels around the circle
    for label, theta in zip(ax.get_xticklabels(), theta):
        if theta in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < theta < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
    
    ax.set_ylim(YLIM[0], YLIM[1])
    ax.tick_params(axis="y", labelsize=LABEL_SIZE_Y)
    ax.tick_params(axis="x", labelsize=LABEL_SIZE_X)
    ax.set_rlabel_position(180 / len(RATINGS))
    if LABELS is not None:
        ax.legend(bbox_to_anchor=(1.42,1.08), prop={'size': LEGEND_SIZE})
    
    if noticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False, axis="y")
    
    ax.set_title(title, fontsize=TITLE_SIZE)
    
    if save is not None:
        plt.savefig(save, format="png", bbox_inches="tight", transparent=True)
    
    plt.show()

def compute_all(path_1, path_2, path_save):
    # path_1 = "./static/images/mat01_1.jpg"
    image_1 = open_vid_image(path_1)
    print("Image 1 opened")
    # image_1 = image_1[2035:2390, 1260:1635]

    # path_2 = "./static/images/mat01_2.jpg"
    image_2 = open_vid_image(path_2)
    print("Image 2 opened")
    # image_2 = image_2[2020:2355, 1235:1645]

    print("Computation start")
    res = predict_ratings_changed_photo(image_1, image_2)
    # plot_res([res["PREDICT_RATINGS"]], colors=["blue"], labels=["Predicted PHOTO"])
    print("Computation end")
    polar_plot([res["PREDICT_RATINGS"]],
           RATINGS=RATING_NAMES, COLORS=["blue"],
           LABELS=["Model predictions - Photos"],
           order=RATING_CHANGE, save=path_save)
    print("Polar plot saved")