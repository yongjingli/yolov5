# 加入size过滤

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        # size filter
        b_size_filter = True
        if b_size_filter and len(labels) > 0:
            labels_tmp = labels.copy()

            labels_tmp[:, 1:5] = xyxy2xywh(labels_tmp[:, 1:5])  # convert xyxy to xywh
            labels_tmp[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels_tmp[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

            b_w = labels_tmp[:, 3] > 0.05
            b_h = labels_tmp[:, 4] > 0.05
            b_filter = [b_w[i] and b_h[i] for i in range(len(b_w))]
            labels = labels[b_filter, :]

        nL = len(labels)  # number of labels
