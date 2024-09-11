# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import scipy as sp
from matplotlib import pyplot as plt
from PIL import Image

"""
def pyramid_dist(i1, i2, rad):
    print("______________________________________________________________________")
    height, width = i1.shape
    if max(height, width) < 400:
        pyramid_dist_out = align_dist(i1, i2, 15, [15, 15])
    else:
        i1half = np.array(Image.fromarray(255 * i1).resize((width // 2, height // 2), Image.Resampling.LANCZOS)) / 255
        i2half = np.array(Image.fromarray(255 * i2).resize((width // 2, height // 2), Image.Resampling.LANCZOS)) / 255
        pyramid_dist_out = align_dist(i1, i2, rad, 2 * pyramid_dist(i1half, i2half, rad))
    return pyramid_dist_out

def pyramid_dist(i1, i2):
    allImages = []
    allImages.append([i1, i2])
    height, width = i1.shape
    while True:
        if max(height, width) < 400:
            break
        else:
            i1half = np.array(Image.fromarray((255 * i1)).resize((width // 2, height // 2), Image.Resampling.LANCZOS)) / 255
            i2half = np.array(Image.fromarray((255 * i2)).resize((width // 2, height // 2), Image.Resampling.LANCZOS)) / 255
            allImages.append([i1half, i2half])
            i1, i2 = i1half, i2half
            height /= 2
            width /= 2
    #img2 = allImages[len(allImages) - 1][1]
    for i in range(len(allImages) - 1):
        img1 = allImages[len(allImages) - 1 - i][0]
        img2 = allImages[len(allImages) - 1 - i][1]
        new_align_dist, _, _ = sk.registration.phase_cross_correlation(img1, img2)
        new_align_dist *= 2
        allImages[len(allImages) - i][1] = np.roll(np.roll(allImages[len(allImages) - i][1], int(new_align_dist[0]), axis = 0), int(new_align_dist[1]), axis = 1)
    #new_align_dist, _, _ = sk.registration.phase_cross_correlation(img1, img2)
    return allImages[len(allImages) - i][1]

def pyramid_dist(i1, i2, curr_dist):
    print("______________________________________________________________________")
    height, width = i1.shape
    if max(height, width) < 400:
        pyramid_dist_out = align_dist(i1, i2, [0, 0])
    else:
        i1half = np.array(Image.fromarray(255 * i1).resize((width // 2, height // 2), Image.Resampling.LANCZOS)) / 255
        i2half = np.array(Image.fromarray(255 * i2).resize((width // 2, height // 2), Image.Resampling.LANCZOS)) / 255
        pyramid_dist_out = align_dist(i1, i2, pyramid_dist(i1half, i2half, curr_dist))
    return pyramid_dist_out

def align_dist(i1, i2, curr_dist):
    new_align_dist, _, _ = sk.registration.phase_cross_correlation(i1, i2)
    for i in range(len(curr_dist)):
        new_align_dist[i] += curr_dist[i]
    return new_align_dist

def align_dist(i1, i2, rad, bnd):
    max = 0
    align_dist = 0
    height, width = i1.shape
    #i1norm = (i1/np.linalg.norm(i1))
    i1crop = i1[int(0.4 * height): int(0.6 * height), int(0.4 * width): int(0.6 * width)]
    i1norm = (i1crop/np.linalg.norm(i1crop))
    for xpos in range(bnd[0] - rad, bnd[0] + rad):
        for ypos in range(bnd[1] - rad, bnd[1] + rad):
            print("loop_" + str(xpos) + "_" + str(ypos))
            #img1 = i1
            img2 = np.roll(np.roll(i2, xpos, axis = 0), ypos, axis = 1)
            #curr_cc_score = ((img1/np.linalg.norm(img1)) * (img2/np.linalg.norm(img2))).ravel().sum()
            img2crop = img2[int(0.4 * height): int(0.6 * height), int(0.4 * width): int(0.6 * width)]
            #curr_cc_score = (i1norm * (img2/np.linalg.norm(img2))).ravel().sum()
            curr_cc_score = (i1norm * (img2crop/np.linalg.norm(img2crop))).ravel().sum()
            if curr_cc_score > max:
                align_dist = [xpos, ypos]
                max = curr_cc_score
    return align_dist

def align_dist(i1, i2, rad, bnd):
    max = 1e20
    align_dist = 0
    height, width = i1.shape
    #i1norm = (i1/np.linalg.norm(i1))
    i1crop = i1[int(0.3 * height): int(0.7 * height), int(0.3 * width): int(0.7 * width)]
    i1norm = (i1crop/np.linalg.norm(i1crop))
    for xpos in range(bnd[0] - rad, bnd[0] + rad):
        for ypos in range(bnd[1] - rad, bnd[1] + rad):
            print("loop_" + str(xpos) + "_" + str(ypos))
            #img1 = i1
            img2 = np.roll(np.roll(i2, xpos, axis = 0), ypos, axis = 1)
            #curr_cc_score = ((img1/np.linalg.norm(img1)) * (img2/np.linalg.norm(img2))).ravel().sum()
            img2crop = img2[int(0.3 * height): int(0.7 * height), int(0.3 * width): int(0.7 * width)]
            #curr_cc_score = (i1norm * (img2/np.linalg.norm(img2))).ravel().sum()
            #curr_cc_score = (i1norm * (img2crop/np.linalg.norm(img2crop))).ravel().sum()
            curr_cc_score = np.sum((i1crop - img2crop) ** 2)
            if curr_cc_score < max:
                align_dist = [xpos, ypos]
                max = curr_cc_score
    return align_dist
"""

# name of the input file
#imname = 'cathedral.jpg'
imname = 'emir.tif'
#imname = 'tobolsk.jpg'
#imname = 'monastery.jpg'
imgnames = ['cathedral.jpg', 'church.tif', 'emir.tif', 'harvesters.tif', 'icon.tif', 'lady.tif', 'melons.tif', 'monastery.jpg', 'onion_church.tif', 'sculpture.tif', 'self_portrait.tif', 'three_generations.tif', 'tobolsk.jpg', 'train.tif']

for imname in imgnames:
    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int_)
    width = im.shape[1]

    # separate color channels
    #b = im[:height]
    #g = im[height: 2*height]
    #r = im[2*height: 3*height]

    factor = 0.075

    chbottom = int(factor * height)
    chtop = int((1 - factor) * height)
    cwleft = int(factor * width)
    cwright = int((1 - factor) * width)

    print("chbottom: " + str(chbottom))
    print("chtop: " + str(chtop))
    print("cwleft: " + str(cwleft))
    print("cwright: " + str(cwright))

    b = im[chbottom: chtop, cwleft: cwright]
    g = im[height + chbottom: height + chtop, cwleft: cwright]
    r = im[height + height + chbottom: height + height + chtop, cwleft: cwright]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    #align_dist_g = sk.registration.phase_cross_correlation(b, g)
    #align_dist_r = sk.registration.phase_cross_correlation(b, r)

    #b -= 0 - np.min(b)
    #b += 1 - np.max(b)
    #g -= 0 - np.min(g)
    #g += 1 - np.max(g)
    #r -= 0 - np.min(r)
    #r += 1 - np.max(r)

    #images = [b, g, r]

    #for image in images:
    #    for y in range(chtop - chbottom):
    #        for x in range(cwright - cwleft):
    #            if image[y][x] > 0.95:
    #                image[y][x] = 1
    #            else:
    #                image[y][x] = 0

    #skio.imshow(b)
    #skio.show()
    #skio.imshow(g)
    #skio.show()
    #skio.imshow(r)
    #skio.show()

    #b_sobel = np.sqrt(sp.ndimage.sobel(b, 0)**2 + sp.ndimage.sobel(b, 1)**2)
    #g_sobel = np.sqrt(sp.ndimage.sobel(g, 0)**2 + sp.ndimage.sobel(g, 1)**2)
    #r_sobel = np.sqrt(sp.ndimage.sobel(r, 0)**2 + sp.ndimage.sobel(r, 1)**2)

    """
    images = [b_sobel, g_sobel, r_sobel]

    for image in images:
        for y in range(chtop - chbottom):
            for x in range(cwright - cwleft):
                if image[y][x] > 0.05:
                    image[y][x] = 1
                else:
                    image[y][x] = 0
    """

    #b_sobel = np.sqrt(sp.ndimage.sobel(b * 255, 0)**2 + sp.ndimage.sobel(b * 255, 1)**2)
    #b_sobel *= 255.0 / np.max(b_sobel)
    #g_sobel = np.sqrt(sp.ndimage.sobel(g * 255, 0)**2 + sp.ndimage.sobel(g * 255, 1)**2)
    #b_sobel *= 255.0 / np.max(g_sobel)
    #r_sobel = np.sqrt(sp.ndimage.sobel(r * 255, 0)**2 + sp.ndimage.sobel(r * 255, 1)**2)
    #b_sobel *= 255.0 / np.max(r_sobel)

    #skio.imshow(b_sobel)
    #skio.show()
    #skio.imshow(g_sobel)
    #skio.show()
    #skio.imshow(r_sobel)
    #skio.show()

    """
    if imname[-4:] == '.tif':
        align_dist_g = pyramid_dist(b_sobel, g_sobel, 20)
        align_dist_r = pyramid_dist(b_sobel, r_sobel, 20)
    else:
        align_dist_g = align_dist(b_sobel, g_sobel, 20, [20, 20])
        align_dist_r = align_dist(b_sobel, r_sobel, 20, [20, 20])
    """

    """
    if imname[-4:] == '.tif':
        align_dist_g = pyramid_dist(b, g, 10)
        align_dist_r = pyramid_dist(b, r, 10)
    else:
        align_dist_g = align_dist(b, g, 20, [20, 20])
        align_dist_r = align_dist(b, r, 20, [20, 20])
    """

    if imname[-4:] == '.tif':
        b_8 = np.array(Image.fromarray(255 * b).resize((width // 4, height // 4), Image.Resampling.LANCZOS)) / 255
        g_8 = np.array(Image.fromarray(255 * g).resize((width // 4, height // 4), Image.Resampling.LANCZOS)) / 255
        r_8 = np.array(Image.fromarray(255 * r).resize((width // 4, height // 4), Image.Resampling.LANCZOS)) / 255
        #b_4 = np.array(Image.fromarray(255 * b).resize((width // 4, height // 4), Image.Resampling.LANCZOS)) / 255
        #g_4 = np.array(Image.fromarray(255 * g).resize((width // 4, height // 4), Image.Resampling.LANCZOS)) / 255
        #r_4 = np.array(Image.fromarray(255 * r).resize((width // 4, height // 4), Image.Resampling.LANCZOS)) / 255
        #b_2 = np.array(Image.fromarray(255 * b).resize((width // 2, height // 2), Image.Resampling.LANCZOS)) / 255
        #g_2 = np.array(Image.fromarray(255 * g).resize((width // 2, height // 2), Image.Resampling.LANCZOS)) / 255
        #r_2 = np.array(Image.fromarray(255 * r).resize((width // 2, height // 2), Image.Resampling.LANCZOS)) / 255

        align_dist_g_8 = sk.registration.phase_cross_correlation(b_8, g_8)[0]
        align_dist_r_8 = sk.registration.phase_cross_correlation(b_8, r_8)[0]
        #align_dist_g_8 *= 2
        #align_dist_r_8 *= 2
        #print(str(align_dist_g_8))
        #g_4 = np.roll(np.roll(g_4, int(align_dist_g_8[0]), axis = 0), int(align_dist_g_8[1]), axis = 1)
        #r_4 = np.roll(np.roll(r_4, int(align_dist_r_8[0]), axis = 0), int(align_dist_r_8[1]), axis = 1)
        #align_dist_g_4 = sk.registration.phase_cross_correlation(b_4, g_4)[0]
        #align_dist_r_4 = sk.registration.phase_cross_correlation(b_4, r_4)[0]
        #align_dist_g_4 *= 2
        #align_dist_r_4 *= 2
        #g_2 = np.roll(np.roll(g_2, int(align_dist_g_4[0]), axis = 0), int(align_dist_g_4[1]), axis = 1)
        #r_2 = np.roll(np.roll(r_2, int(align_dist_r_4[0]), axis = 0), int(align_dist_r_4[1]), axis = 1)
        #align_dist_g_2 = sk.registration.phase_cross_correlation(b_2, g_2)[0]
        #align_dist_r_2 = sk.registration.phase_cross_correlation(b_2, r_2)[0]
        #align_dist_g_2 *= 2
        #align_dist_r_2 *= 2
        #g_1 = np.roll(np.roll(g, int(align_dist_g_2[0]), axis = 0), int(align_dist_g_4[1]), axis = 1)
        #r_1 = np.roll(np.roll(r, int(align_dist_r_2[0]), axis = 0), int(align_dist_r_4[1]), axis = 1)
        #align_dist_g = sk.registration.phase_cross_correlation(b, g_1)[0]
        #align_dist_r = sk.registration.phase_cross_correlation(b, r_1)[0]
        align_dist_g = align_dist_g_8 * 4
        align_dist_r = align_dist_r_8 * 4
    else:
        align_dist_g, _, _ = sk.registration.phase_cross_correlation(b, g)
        align_dist_r, _, _ = sk.registration.phase_cross_correlation(b, r)

    #align_dist_g = pyramid_dist(b, g, 10)
    #align_dist_r = pyramid_dist(b, r, 10)

    #align_dist_g = pyramid_dist(b_sobel, g_sobel, 10)
    #align_dist_r = pyramid_dist(b_sobel, r_sobel, 10)

    #align_dist_g = align_dist(b, g, 20, [20, 20])
    #align_dist_r = align_dist(b, r, 20, [20, 20])

    #align_dist_g, _, _ = sk.registration.phase_cross_correlation(b, g)
    #align_dist_r, _, _ = sk.registration.phase_cross_correlation(b, r)

    #align_dist_g = pyramid_dist(b, g, [0, 0])
    #align_dist_r = pyramid_dist(b, r, [0, 0])

    # final alignment before layering images
    #ag = np.roll(np.roll(g, align_dist_g[0], axis = 0), align_dist_g[1], axis = 1)
    #ar = np.roll(np.roll(r, align_dist_r[0], axis = 0), align_dist_r[1], axis = 1)

    ag = np.roll(np.roll(g, int(align_dist_g[0]), axis = 0), int(align_dist_g[1]), axis = 1)
    ar = np.roll(np.roll(r, int(align_dist_r[0]), axis = 0), int(align_dist_r[1]), axis = 1)

    #ag = pyramid_dist(b, g)
    #ar = pyramid_dist(b, r)

    #### ag = align(g, b)
    #### ar = align(r, b)
    # create a color image
    im_out = np.dstack([ar, ag, b])
    #im_out = np.dstack([r, g, b])

    # convert to file output friendly values
    ag_out = Image.fromarray((ag * 255).astype(np.uint8))
    ar_out = Image.fromarray((ar * 255).astype(np.uint8))
    b_out = Image.fromarray((b * 255).astype(np.uint8))
    im_out_file = np.dstack([ar_out, ag_out, b_out])

    # save the image
    fname = imname[:-4] + '_out.jpg'
    #fname = 'img_outputs_final/' + imname[:-4] + '_out' + imname[-4:]
    skio.imsave(fname, im_out_file)

    # display the image
    skio.imshow(im_out)
    skio.show()
