import glob
from tools.test import *
import cv2
import os
import run_ssd_live_demo as person

vidcap = cv2.VideoCapture('../../data/smallv1.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        return hasFrames, image
    return hasFrames, None

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/video', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    # img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    # ims = [cv2.imread(imf) for imf in img_files]

    sec = 0
    frameRate = 0.8 #//it will capture image in each 0.5 second
    count=0
    success, img = getFrame(sec)
    height , width , layers =  img.shape
    print(width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('video.avi',fourcc, 20.0, (2560, 1600))
    isFirstImage = True
    toc = 0
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success, img = getFrame(sec)
        print(success)
        if success:
            tic = cv2.getTickCount()
            if isFirstImage:
                isFirstImage = False
                # cv2.namedWindow("Unacademy demo tutorial", cv2.WND_PROP_FULLSCREEN)
                x, y, w, h = person.PersonDetector(img)
            if count == 1:
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                state = siamese_init(img, target_pos, target_sz, siammask, cfg['hp'], device=device)
            else:
                state = siamese_track(state, img, mask_enable=True, refine_enable=True, device=device)  # track
                location = state['ploygon'].flatten()
                mask = state['mask'] > state['p'].seg_thr

                img[:, :, 2] = (mask > 0) * img[:, :,2] + (mask == 0) * 255
                img[:, :, 0] = (mask > 0) * img[:, :,0] + (mask == 0) * 255
                img[:, :, 1] = (mask > 0) * img[:, :,1] + (mask == 0) * 255
                height, weight, layers = img.shape
                print("size", width, height)
                # cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                # cv2.imshow('Unacademy demo tutorial', img)
                video.write(img)
                key = cv2.waitKey(1)
                if key > 0:
                    break
            temp = cv2.getTickCount() - tic
            toc += temp
            temp1 = temp/cv2.getTickFrequency()
            print("frame " + str(count) + "   fps: " + str(1 / temp1))

        toc /= cv2.getTickFrequency()
        fps = count / toc
        video.release()
        print('Unacademy demo tutorial Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))