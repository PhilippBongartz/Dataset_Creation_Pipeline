import cv2
import matplotlib.pyplot as plt
from cv2 import CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES
from src.utils import *



@overall_runtime
def read_the_nth_frame(path, n):
    cap = cv2.VideoCapture(path)
    cap.set(CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    cap.release()
    return frame

@overall_runtime
def read_every_nth_frame(path, n):
    cap = cv2.VideoCapture(path,cv2.CAP_FFMPEG) 
    success = cap.grab()
    fno = 0
    while success:
        if fno%n == 0:
            try:
                _,frame = cap.retrieve()
                yield frame
            except cv2.error as e:
                print("OpenCV error: ", e)
            except Exception as e:
                print("Unexpected error: ", e)
        fno+=1
        success = cap.grab()
    cap.release()
        	

@overall_runtime
def read_every_nth_frame_old(path, n):
    cap = cv2.VideoCapture(path)
    total_frames = cap.get(CAP_PROP_FRAME_COUNT)

    for t in range(0, int(total_frames), n):
        cap.set(CAP_PROP_POS_FRAMES, t)
        ret, frame = cap.read()
        yield frame
    cap.release()
    
# CV2 Videoframes lesen, verarbeitung und plotten

@overall_runtime
def count_frames(path):
    cap = cv2.VideoCapture(path)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    frames = []
    count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count += 1
        else:
            break
    cap.release()
    return count
    
@overall_runtime
def get_frames_per_second(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

@overall_runtime
def frame_number_to_timestamp(frame_number,fps):
    return frame_number / fps
    
@overall_runtime
def get_comments_per_timeintervall(transcript,start,end):
    """nicht effizient aber für's erste reichts"""
    comment = ''
    for entry in transcript:
        if 'duration' in entry:
            if entry['start']+entry['duration']>start and entry['start']<end:
                comment += entry['text'] + ' '
        if 'end' in entry:
            if entry['end']>start and entry['start']<end:
                comment += entry['text'] + ' '
    return comment

@overall_runtime
def read_frames(path, von, bis):
    cap = cv2.VideoCapture(path)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    frames = []
    count = 0
    while (cap.isOpened()):
        count += 1
        if count > bis:
            break

        ret, frame = cap.read()
        if ret == True:
            if count >= von and count < bis:
                frames.append(frame)
        else:
            break
    cap.release()
    return frames


@overall_runtime
def iterate_over_frames(path, von=0, bis=1000000000):
    cap = cv2.VideoCapture(path)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    count = 0
    while (cap.isOpened()):
        count += 1
        if count > bis:
            break

        ret, frame = cap.read()
        if ret == True:
            if count >= von and count < bis:
                yield frame
        else:
            break
    cap.release()
    
@overall_runtime
def graying(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

@overall_runtime
def rgbing(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

@overall_runtime
def plotting(img, title='img'):
    plt.imshow(img)
    plt.title(title)
    plt.show()
