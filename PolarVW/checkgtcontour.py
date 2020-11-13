slicei = -1
sid = caseloader.getsid(slicei)

gtbb = caseloader.load_gt_bb(slicei)

cart_patch = cv2.resize(croppatch(dicomstack[sid], gtbb.y, gtbb.x, 64, 64), (512, 512))
cart_patch = np.repeat(cart_patch[:, :, None], 3, axis=2)
cart_patch = cart_patch / np.max(cart_patch)

for cti in range(len(contours[0])):
    ctx = int(round((contours[0][cti][0] - gtbb.x) * SCALE + 256))
    cty = int(round((contours[0][cti][1] - gtbb.y) * SCALE + 256))
    cart_patch[cty, ctx, 0] = np.max(cart_patch)

    ctx = int(round((contours[1][cti][0] - gtbb.x) * SCALE + 256))
    cty = int(round((contours[1][cti][1] - gtbb.y) * SCALE + 256))
    cart_patch[cty, ctx, 0] = np.max(cart_patch)

cart_cont = caseloader.load_cart_cont(slicei)

for cti in range(len(cart_cont[0])):
    ctx = int(round((cart_cont[0][cti][0] / SCALE - gtbb.x) * SCALE + 256))
    cty = int(round((cart_cont[0][cti][1] / SCALE - gtbb.y) * SCALE + 256))
    cart_patch[cty, ctx, 1] = np.max(cart_patch)
for cti in range(len(cart_cont[1])):
    ctx = int(round((cart_cont[1][cti][0] / SCALE - gtbb.x) * SCALE + 256))
    cty = int(round((cart_cont[1][cti][1] / SCALE - gtbb.y) * SCALE + 256))
    cart_patch[cty, ctx, 1] = np.max(cart_patch)

plt.figure(figsize=(10, 10))
plt.imshow(cart_patch)

# check pred contour agrees with pred label
segvw = plotct(512 * SCALE, contourin * SCALE, contourout * SCALE)
cart_vw_seg = croppatch(segvw, gtbb.y * SCALE, gtbb.x * SCALE, 256, 256)
cart_patch = np.repeat(cart_vw_seg[:, :, None], 3, axis=2)
cart_patch[:, :, 1] = 0
cart_patch[:, :, 2] = 0
cart_patch = cart_patch / np.max(cart_patch)

for cti in range(len(contour_in_roi)):
    ctx = int(round(contour_in_roi[cti][0]))
    cty = int(round(contour_in_roi[cti][1]))
    cart_patch[cty, ctx, 1] = np.max(cart_patch)
for cti in range(len(contour_out_roi)):
    ctx = int(round(contour_out_roi[cti][0]))
    cty = int(round(contour_out_roi[cti][1]))
    cart_patch[cty, ctx, 1] = np.max(cart_patch)

plt.figure(figsize=(10, 10))
plt.imshow(cart_patch)

# check gt contour agrees with pred label
cart_label = caseloader.load_cart_vw(slicei)
cart_patch = np.repeat(cart_label[:, :, None], 3, axis=2)
cart_patch[:, :, 1] = 0
cart_patch[:, :, 2] = 0
cart_patch = cart_patch / np.max(cart_patch)

for cti in range(len(cart_gt_cont_in_roi)):
    ctx = int(round(cart_gt_cont_in_roi[cti][0]))
    cty = int(round(cart_gt_cont_in_roi[cti][1]))
    cart_patch[cty, ctx, 1] = np.max(cart_patch)
for cti in range(len(cart_gt_cont_out_roi)):
    ctx = int(round(cart_gt_cont_out_roi[cti][0]))
    cty = int(round(cart_gt_cont_out_roi[cti][1]))
    cart_patch[cty, ctx, 1] = np.max(cart_patch)

plt.figure(figsize=(10, 10))
plt.imshow(cart_patch)
