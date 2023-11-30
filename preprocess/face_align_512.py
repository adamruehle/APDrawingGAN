import cv2
import numpy as np
import os

def face_align_512(impath, facial5point, imglocation, txtlocation):
    # Alignment settings
    imgSize = (512, 512)
    coord5point = np.array([
        [180, 230],
        [300, 230],
        [240, 301],
        [186, 365.6],
        [294, 365.6]
    ])
    coord5point = ((coord5point - 240) / 560) * 512 + 256

    # Face alignment
    img = cv2.imread(impath)
    xCoords = facial5point['landmarks'][:, 0]
    yCoords = facial5point['landmarks'][:, 1]
    # Face alignment (continued)
    facial5point = np.column_stack((xCoords, yCoords))
    transf = cv2.estimateAffinePartial2D(facial5point, coord5point)[0]

    # Get the transformation matrix for similarity
    transf = np.vstack([transf, [0, 0, 1]])
    transf[2] = [0, 0, 1]

    # Apply transformation to the image
    rows, cols, _ = img.shape
    trans_img = cv2.warpAffine(img, transf[:2], (cols, rows), borderValue=(255, 255, 255))

    # Crop to 512x512 at the top left corner
    trans_img = trans_img[:512, :512]

    # Round the transformed facial landmarks
    homogenous_points = np.column_stack((facial5point, np.ones(facial5point.shape[0])))
    homogenous_transf = np.dot(transf, homogenous_points.T).T
    trans_facial5point = np.round(homogenous_transf[:, :2]).astype(int)

    # Save results
    name = os.path.splitext(os.path.basename(impath))[0]
    cv2.imwrite(f"{imglocation}/{name}_aligned.png", trans_img)
    write_5pt(f"{txtlocation}/{name}_aligned.txt", trans_facial5point)

    # Return the aligned image path
    return f"{imglocation}/{name}_aligned.png"

def write_5pt(fn, trans_pt):
    with open(fn, 'w') as f:
        for pt in trans_pt:
            f.write(f"{pt[0]} {pt[1]}\n")
