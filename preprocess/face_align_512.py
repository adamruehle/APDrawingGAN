import cv2
import numpy as np

def face_align_512(impath, facial5point, savedir):
    # Alignment settings
    imgSize = (512, 512)
    coord5point = np.array([[180, 230],
                            [300, 230],
                            [240, 301],
                            [186, 365.6],
                            [294, 365.6]])

    # Convert coordinates to match Python indexing (0-based)
    coord5point = (coord5point - 240) / 560 * 512 + 256

    # Face alignment
    img = cv2.imread(impath)
    facial5point = np.array(facial5point['landmarks'], dtype=np.float32)
    transf = cv2.estimateAffinePartial2D(facial5point, coord5point)[0]
    trans_img = cv2.warpAffine(img, transf, imgSize, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    trans_facial5point = np.round(cv2.transform(facial5point.reshape(1, -1, 2), transf))[0]

    # Save results
    if not cv2.os.path.exists(savedir):
        cv2.os.mkdir(savedir)

    _, name = cv2.os.path.splitext(cv2.os.path.basename(impath))
    name = name[1:]  # Remove leading underscore
    # Save trans_img
    cv2.imwrite(cv2.os.path.join(savedir, f'{name}_aligned.png'), trans_img)
    print(f'Write aligned image to {cv2.os.path.join(savedir, f"{name}_aligned.png")}')

    # Save trans_facial5point
    write_5pt(cv2.os.path.join(savedir, f'{name}_aligned.txt'), trans_facial5point)
    print(f'Write transformed facial landmark to {cv2.os.path.join(savedir, f"{name}_aligned.txt")}')

    # Show results
    cv2.imshow('Aligned Image', trans_img)
    cv2.plot(trans_img, trans_facial5point[:, 0], trans_facial5point[:, 1], color='b')
    cv2.plot(trans_img, trans_facial5point[:, 0], trans_facial5point[:, 1], 'r+', markersize=10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_5pt(fn, trans_pt):
  with open(fn, 'w') as fid:
    for i in range(5):
      fid.write(f'{int(trans_pt[i, 0])} {int(trans_pt[i, 1])}\n')