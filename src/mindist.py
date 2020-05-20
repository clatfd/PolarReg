import numpy as np
import matplotlib.pyplot as plt
from src.UTL import croppatch,fillpatch
from scipy.ndimage import gaussian_filter
import copy
import cv2


def min_dist(ptx, pty, mask, direction):
    deltax = np.cos(direction / 180 * np.pi)
    deltay = np.sin(direction / 180 * np.pi)
    dist = 0
    cval = mask[pty, ptx]
    while cval > 0:
        ptx += deltax
        pty += deltay
        dist += 1
        cval = mask[int(round(pty)), int(round(ptx))]

    return dist


def min_dist_pt(ptx, pty, mask):
    cdist = np.inf
    for di in range(0, 360, 45):
        cdist = min(cdist, min_dist(ptx, pty, mask, di))
    return cdist


def get_min_dist_map(mask):
    min_dist_map = []
    deltax = [1, 1, 0, -1, -1, -1, 0, 1]
    deltay = [0, 1, 1, 1, 0, -1, -1, -1]
    for i in range(len(deltax)):
        # print('dir',deltax[i],deltay[i])
        min_dist_map.append(get_min_dist_dir_map(mask, deltax[i], deltay[i]))
    return np.moveaxis(np.array(min_dist_map), 0, 2)


def get_min_dist_dir_map(mask, deltax, deltay):
    min_dist_map = np.zeros(mask.shape)
    maskregion = np.where(mask > 0)
    min_dist_map[maskregion] = -1
    # deltax = int(np.ceil(np.cos(direction / 180 * np.pi)))
    # deltay = int(np.ceil(np.sin(direction / 180 * np.pi)))
    delta = np.sqrt(deltax * deltax + deltay * deltay)
    for pi in range(maskregion[0].shape[0]):
        cpty = maskregion[0][pi]
        cptx = maskregion[1][pi]
        if min_dist_map[cpty][cptx] == -1:
            # print('pt',cpty,cptx)
            # extract one line
            step = 0
            extx = cptx
            exty = cpty
            steps = []
            ptxy = []
            while step < mask.shape[0] * 1.42 and min_dist_map[exty][extx] == -1:
                ptxy.append([extx, exty])
                steps.append(step)
                step += 1
                extx = cptx + deltax * step
                exty = cpty + deltay * step

            largeststep = min_dist_map[exty][extx] + step * delta
            # print(extx,exty,min_dist_map[exty][extx],largeststep)
            # paint min_dist_map along the line
            for cstepi in range(len(steps)):
                cstep = steps[cstepi]
                cdist = largeststep - cstep * delta
                extx = ptxy[cstepi][0]
                exty = ptxy[cstepi][1]
                min_dist_map[exty][extx] = cdist
                # print('pt', extx, exty, cdist)

    return min_dist_map


def find_nms_center(dist_map, fig=0):
    rad = 3
    nms_dist_map = copy.copy(dist_map)
    nms_dist_map = gaussian_filter(nms_dist_map, sigma=3)
    if fig:
        nms_dist_map_p = np.zeros((nms_dist_map.shape[0], nms_dist_map.shape[1], 3), dtype=np.uint8)
        nms_dist_map_p[:, :, 0] = dist_map / np.max(dist_map) * 255

    thres = np.max(nms_dist_map) * 0.8
    nz = np.where(dist_map > thres)
    min_ct_dist_thres = 40
    nms_centers = {}
    for pi in range(nz[0].shape[0]):
        pty = nz[0][pi]
        ptx = nz[1][pi]
        if pty < rad or ptx < rad:
            continue
        mn = np.max(nms_dist_map[pty - rad:pty + rad + 1, ptx - rad:ptx + rad + 1])
        if nms_dist_map[pty, ptx] == mn:
            has_nei_ct_higher = False
            for cti in list(nms_centers.keys()):
                eptx = int(cti.split('-')[0])
                epty = int(cti.split('-')[1])
                ethres = nms_centers[cti]
                if pow(eptx - ptx, 2) + pow(eptx - ptx, 2) < pow(min_ct_dist_thres, 2):
                    if ethres > mn:
                        print(ptx, pty, mn, 'smaller than', cti, ethres)
                        has_nei_ct_higher = True
                    else:
                        nms_centers['%d-%d' % (ptx, pty)] = mn
                        del nms_centers[cti]
                #else:
                #    print(np.sqrt(pow(eptx - ptx, 2) + pow(eptx - ptx, 2)), 'over', min_ct_dist_thres)
            if has_nei_ct_higher == False:
                nms_centers['%d-%d' % (ptx, pty)] = mn

    # if fig:
    #    plt.imshow(nms_dist_map_p / np.max(nms_dist_map_p))
    #    plt.show()
    if fig:
        for cti in nms_centers.keys():
            ptx = int(cti.split('-')[0])
            pty = int(cti.split('-')[1])
            nms_dist_map_p[pty - 5:pty + 6, ptx - 1:ptx + 2, 1] = 255
            nms_dist_map_p[pty - 1:pty + 2, ptx - 5:ptx + 6, 1] = 255
    if fig:
        return nms_centers, nms_dist_map_p
    else:
        return nms_centers


# find center
def gen_2d_gaussion(size, sigma=1, mu=0):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g


# predict multiple min dist map and average
def multi_min_dist_map(min_dist_cnn, dicomslice, octy, octx, patchheight, patchwidth):
    search_patch_rz_batch = []
    stride = 10
    rg = 2
    for ofi in np.arange(-rg, rg + 1):
        for ofj in np.arange(-rg, rg + 1):
            octyof = octy + ofi * stride
            octxof = octx + ofj * stride
            searchpatch = croppatch(dicomslice, octyof, octxof, patchheight, patchwidth)
            searchpatchrz = cv2.resize(searchpatch, (0, 0), fx=4, fy=4)
            searchpatchrz = searchpatchrz/np.max(searchpatchrz)
            # searchpatchrz = searchpatch
            search_patch_rz_batch.append(searchpatchrz)
            # plt.imshow(searchpatchrz)
            # plt.title('Original Patch'+str(octyof)+str(octxof))
            # plt.show()

    search_patch_rz_batch = np.array(search_patch_rz_batch)
    min_dist_search_patch_batch = min_dist_cnn.predict(search_patch_rz_batch[:, :, :, None])[:, :, :, 0]
    min_dist_search_patch = []
    for i in range(min_dist_search_patch_batch.shape[0]):
        ofi = i // (2 * rg + 1) - rg
        ofj = i % (2 * rg + 1) - rg
        # print(ofi,ofj)
        offpred = croppatch(min_dist_search_patch_batch[i], 256 - ofi * stride * 4, 256 - ofj * stride * 4, 256,
                            256)
        # offpred = croppatch(min_dist_search_patch_batch[i], 256 - ofi * stride, 256 - ofj * stride, 256, 256)
        '''plt.title('Original Patch'+str(ofi)+str(ofj))
        plt.imshow(offpred)
        plt.colorbar()
        plt.show()
        print(np.max(offpred))'''
        min_dist_search_patch.append(offpred / np.max(offpred))
    min_dist_search_patch = np.array(min_dist_search_patch)
    min_dist_search_patch = np.max(min_dist_search_patch, axis=0)
    # plt.imshow(min_dist_search_patch)

    return min_dist_search_patch

def multi_min_dist_pred(min_dist_cnn, dicomslice, octy, octx, patchheight, patchwidth, kernelmask=None):
    min_dist_search_patch = multi_min_dist_map(min_dist_cnn, dicomslice, octy, octx, patchheight, patchwidth)
    if kernelmask is None:
        kernelmask = gen_2d_gaussion(min_dist_search_patch.shape[0], 0.5)
    dist_pred_ct, nms_dist_map_p = find_nms_center(min_dist_search_patch * kernelmask, fig=1)
    search_patch_center = croppatch(dicomslice, octy, octx, patchheight, patchwidth)
    search_patch_center_rz = cv2.resize(search_patch_center, (0, 0), fx=4, fy=4)
    # search_patch_center_rz = search_patch_center
    nms_dist_map_p[:, :, 2] = search_patch_center_rz / np.max(search_patch_center_rz) * 255

    cts = [[int(i.split('-')[0]), int(i.split('-')[1])] for i in list(dist_pred_ct.keys())]

    if len(cts) == 0:
        print('no ct')
    elif len(cts) != 1:
        print('cts', len(cts))
    return cts, nms_dist_map_p

def multi_min_dist_pred_withinbb(min_dist_cnn, dicomslice, bb, patchheight, patchwidth):
    SCALE = 4
    octy = bb.y
    octx = bb.x
    min_dist_search_patch = multi_min_dist_map(min_dist_cnn, dicomslice, octy, octx, patchheight, patchwidth)
    min_dist_search_patch_rz = croppatch(min_dist_search_patch,min_dist_search_patch.shape[1]/2,min_dist_search_patch.shape[0]/2,bb.h/2*SCALE,bb.w/2*SCALE)
    min_dist_search_patch_rz = fillpatch(np.zeros(min_dist_search_patch.shape),min_dist_search_patch_rz)
    #plt.imshow(min_dist_search_patch_rz)
    #plt.show()
    dist_pred_ct, nms_dist_map_p = find_nms_center(min_dist_search_patch_rz, fig=1)
    search_patch_center = croppatch(dicomslice, octy, octx, patchheight, patchwidth)
    search_patch_center_rz = cv2.resize(search_patch_center, (0, 0), fx=4, fy=4)
    # search_patch_center_rz = search_patch_center
    nms_dist_map_p[:, :, 2] = search_patch_center_rz / np.max(search_patch_center_rz) * 255

    cts = [[int(i.split('-')[0]), int(i.split('-')[1])] for i in list(dist_pred_ct.keys())]

    if len(cts) == 0:
        print('no ct')
    elif len(cts) != 1:
        print('cts', len(cts))
    return cts, nms_dist_map_p

def find_component_bb(complabel, labelid):
    ypos, xpos = np.where(complabel == labelid)
    xmin = np.min(xpos)
    xmax = np.max(xpos)
    ymin = np.min(ypos)
    ymax = np.max(ypos)
    #min_dist_prob_region = nms_dist_map_p[ymin:ymax,xmin:xmax]
    return BB.fromminmax(xmin, xmax, ymin, ymax)


# find label id from thresholded miin dist map for nms centers, and bb around each connected component
def labelcts(nms_dist_map_p, nms_centers):
    ret, thresh = cv2.threshold(nms_dist_map_p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    connectivity = 4
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    complabel = output[1]
    # plt.imshow(complabel)
    # component label id for each nms centers
    component_label_ids = []
    # map from label id to bb around component
    component_bbs = {}
    for pti in nms_centers:
        component_label_ids.append(complabel[pti[1], pti[0]])
        if complabel[pti[1], pti[0]] not in component_bbs:
            component_bbs[complabel[pti[1], pti[0]]] = find_component_bb(complabel, complabel[pti[1], pti[0]])
    return component_label_ids, component_bbs


from src.BB import BB
def mergects(nms_dist_map_p, nms_centers):
    # threshold and connected component, if cts in the same region, merge
    component_label_ids, component_bbs = labelcts(nms_dist_map_p, nms_centers)
    label_id_cts = {}  # map from label id to array of contours
    for cti in range(len(nms_centers)):
        if component_label_ids[cti] not in label_id_cts:
            label_id_cts[component_label_ids[cti]] = []
        label_id_cts[component_label_ids[cti]].append(nms_centers[cti])
    mcts = []
    for labeli in label_id_cts:
        mct = np.mean(np.array(label_id_cts[labeli]), axis=0).tolist()
        mbb = component_bbs[labeli]
        #print('mbb offset', mbb.x - mct[0], mbb.y - mct[1])
        mcts.append(mbb)
    return mcts


