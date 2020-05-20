import matplotlib.pyplot as plt
import copy
import numpy as np
from src.UTL import dist

def mergecont(contours, nms_centers, method):
    merge_cont = []
    md = []
    median_filt_id = []
    for conti in range(256):
        cdists = [dist(nms_centers[cti], contours[cti][conti]) for cti in range(len(nms_centers))]
        if cdists[0] / cdists[1] > 1.1:
            com_conti = contours[1][conti]
        elif cdists[1] / cdists[0] > 1.1:
            com_conti = contours[0][conti]
        else:
            print('cdists', cdists)
            if len(median_filt_id) == 0 or median_filt_id[-1] == conti - 1:
                median_filt_id.append(conti)
            else:
                md.append(copy.copy(median_filt_id))
                median_filt_id = [conti]

            com_conti = np.sum([contours[cti][conti] / cdists[cti] for cti in range(len(nms_centers))],
                               axis=0) / np.sum([1 / cdists[cti] for cti in range(len(nms_centers))])
        merge_cont.append(com_conti)
    md.append(copy.copy(median_filt_id))
    merge_cont_fil = copy.copy(merge_cont)
    print('md', md)

    for mdi in md:
        if method in ['Mean', 'Median']:
            fz = 10
            for mdii in mdi:
                ci = [merge_cont[mdiii][0] for mdiii in range(mdii - fz // 2, mdii + fz // 2 + 1)]
                co = [merge_cont[mdiii][1] for mdiii in range(mdii - fz // 2, mdii + fz // 2 + 1)]
                if method == 'Mean':
                    merge_cont_fil[mdii] = [np.mean(ci), np.mean(co)]
                elif method == 'Median':
                    merge_cont_fil[mdii] = [np.median(ci), np.median(co)]
        elif method == 'Poly':
            for i in range(10):
                mdi.insert(0, mdi[0] - 1)
                mdi.append(mdi[-1] + 1)

            print(mdi)
            ci = [merge_cont[mdii][0] for mdii in mdi]
            co = [merge_cont[mdii][1] for mdii in mdi]
            # merge_cont_fil[conti] = np.median(md,axis=0)

            z1 = np.polyfit(mdi, ci, 3)
            p1 = np.poly1d(z1)
            z2 = np.polyfit(mdi, co, 3)
            p2 = np.poly1d(z2)
            for mdii in mdi:
                merge_cont_fil[mdii] = [p1(mdii), p2(mdii)]

        plt.plot(np.arange(256), [merge_cont[mdii] for mdii in range(256)])
        plt.show()
        plt.plot(mdi, [merge_cont_fil[mdii] for mdii in mdi])
        plt.show()
        plt.plot(np.arange(256), [merge_cont_fil[mdii] for mdii in range(256)])
        plt.show()
    return merge_cont_fil
