import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from src.utils import *

# Funktionen zur Inner Edge Detektion

@overall_runtime
def kernel_part(size, ones):
    """
    Erstellt einen quadratischen Kernel
    mit einer bestimmten Anzahl von zufällig verteilten 1en.
    """
    kernel = np.zeros(size * size)
    kernel[:ones] = 1
    np.random.shuffle(kernel)
    kernel = kernel.reshape((size, size))
    return kernel

@overall_runtime
def kernel_whole(LO, RO, LU, RU):
    """
    Baut einen Kernel aus vier quadratischen Kerneln zusammen.
    """
    return np.concatenate(
        [np.concatenate([LO, RO], axis=1), np.concatenate([LU, RU], axis=1)],
        axis=0)

@overall_runtime
def inner_edge_kernel(size):
    """
    Ein Kernel der die Quadrate links oben und rechts unten
    mit den Quadraten rechts oben und links unten vergleicht.
    """
    part_size = size // 2

    ones1 = part_size * part_size
    ones2 = part_size * part_size

    LO = kernel_part(part_size, ones2) * -1
    RU = kernel_part(part_size, ones2) * -1

    RO = kernel_part(part_size, ones1)
    LU = kernel_part(part_size, ones1)

    return kernel_whole(LO, RO, LU, RU)

@overall_runtime
def maximum_kernel_image(size, im):
    """
    Verarbeitet ein Bild indem es jeweils das lokale Maximum berechnet.
    """
    size = (size, size)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    max_image = cv2.dilate(im, kernel)
    return max_image

@overall_runtime
def minimum_kernel_image(size, im):
    """
    Verarbeitet ein Bild indem es jeweils das lokalte Minimum berechnet.
    """
    size = (size, size)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    min_image = cv2.erode(im, kernel)
    return min_image

@overall_runtime
def normed_inner_edge_image(gray, size=8):
    """
    Verarbeitet ein Bild indem es lokal genormt inner edges des Schachbretts berechnet.
    """
    if size % 2 != 0:
        print('Size has to be divisible by 2!')
        return

    ie_kernel = inner_edge_kernel(size)

    ie_img = cv2.filter2D(gray.astype('int'), -1, ie_kernel)
    # Durchschnittlicher Differenz der dunklen und hellen Felder
    diff_img = ie_img / 2 * ((size // 2) ** 2)
    max_img = maximum_kernel_image(size, gray)
    min_img = minimum_kernel_image(size, gray)

    # Genormt aus den lokalen Wertebereich
    norm_img = diff_img / (np.abs(max_img - min_img) + np.ones(max_img.shape))  # abs unnötig?

    return norm_img

@overall_runtime
def absolute(img):
    return np.abs(img)


# Funktionen zur Schachbrettextraktion

@overall_runtime
def cutoff_berechnen(abs_norm_img, anzahl):
    # Die Werte
    shape = abs_norm_img.shape
    werte = np.reshape(np.copy(abs_norm_img), (shape[0] * shape[1]))
    werte.sort()

    # Ein Wertcutoff für die Top Feldereckenkernelpunkte
    cutoff = werte[::-1][anzahl]
    return cutoff

@overall_runtime
def punkte_aussuchen(abs_norm_img, cutoff):
    punkte = []
    shape = abs_norm_img.shape
    for t in range(shape[0]):
        for tt in range(shape[1]):
            if abs_norm_img[t, tt] > cutoff:
                punkte.append((t, tt))
    return punkte

@overall_runtime
def alle_rechten_unteren_ecken_fuer_diesen(punkt, feldbreite):
    x, y = punkt
    sign = 1
    for i in range(0, 7):
        x_diff = i * feldbreite + feldbreite
        for j in range(0, 7):
            y_diff = j * feldbreite + feldbreite
            sign *= -1
            yield (round(x + x_diff), round(y + y_diff), sign)

@overall_runtime
def chessboard_voting_finegrained(norm_img, cutoff=1000):
    voting_dict = {}

    punkte = []
    for xx in range(norm_img.shape[0]):
        for yy in range(norm_img.shape[1]):
            punkte.append((np.abs(norm_img[xx, yy]), xx, yy, norm_img[xx, yy]))
    punkte = sorted(punkte)[::-1]

    max_vote = 0
    for i, punkt in enumerate(punkte):
        (abs_wert, xx, yy, wert) = punkt
        if i > cutoff:
            break
        for feldbreite in [fb / 10 for fb in range(200, 1000)]:  # range(20,100):
            for ecke in alle_rechten_unteren_ecken_fuer_diesen((xx, yy), feldbreite):
                x, y, sign = ecke
                if x < norm_img.shape[0] and y < norm_img.shape[1]:
                    if (x, y, feldbreite) in voting_dict:
                        voting_dict[(x, y, feldbreite)] += wert * sign
                        if voting_dict[(x, y, feldbreite)] > max_vote:
                            max_vote = voting_dict[(x, y, feldbreite)]
                    else:
                        voting_dict[(x, y, feldbreite)] = wert * sign

    return sorted(voting_dict.items(), key=lambda item: item[1])[::-1]

@overall_runtime
def chessboard_voting(norm_img, cutoff=1000):
    voting_dict = {}

    punkte = []
    for xx in range(norm_img.shape[0]):
        for yy in range(norm_img.shape[1]):
            punkte.append((np.abs(norm_img[xx, yy]), xx, yy, norm_img[xx, yy]))
    punkte = sorted(punkte)[::-1]

    max_vote = 0
    for i, punkt in enumerate(punkte):
        (abs_wert, xx, yy, wert) = punkt
        if i > cutoff:
            break
        for feldbreite in [fb for fb in range(20, 100)]:  # range(20,100):
            for ecke in alle_rechten_unteren_ecken_fuer_diesen((xx, yy), feldbreite):
                x, y, sign = ecke
                if x < norm_img.shape[0] and y < norm_img.shape[1]:
                    if (x, y, feldbreite) in voting_dict:
                        voting_dict[(x, y, feldbreite)] += wert * sign
                        if voting_dict[(x, y, feldbreite)] > max_vote:
                            max_vote = voting_dict[(x, y, feldbreite)]
                    else:
                        voting_dict[(x, y, feldbreite)] = wert * sign

    return sorted(voting_dict.items(), key=lambda item: item[1])[::-1]

@overall_runtime
def overlapping_interval(iv1, iv2):
    anf1, end1 = iv1
    anf2, end2 = iv2
    if end2 < anf1 or end1 < anf2:
        # print('False',iv1,iv2)
        return False
    # print('True',iv1,iv2)
    return True

@overall_runtime
def overlapping(board1, board2):
    x1, y1, fb1 = board1
    x2, y2, fb2 = board2
    if overlapping_interval((x1 - 8 * fb1, x1), (x2 - 8 * fb2, x2)) and overlapping_interval((y1 - 8 * fb1, y1),
                                                                                             (y2 - 8 * fb2, y2)):
        return True
    return False

@overall_runtime
def remove_overlapping_boards(possible_boards):
    selected = [possible_boards[0]]
    for board, points in possible_boards[1:]:
        overlap = False
        for board2, points2 in selected:
            if overlapping(board, board2):
                # print('dealbreaker',board,board2)
                overlap = True
                break
        if not overlap:
            selected.append((board, points))
            # print('selected',board,points)
        # else:
        # print('ejected ',board,points)
    return selected


# Visualisierung
@overall_runtime
def make_big_point(image,punkt):
    im = copy.deepcopy(image)
    x,y = punkt
    for t in range(-5,6):
        for tt in range(-5,6):
            im[x+t,y+tt] = np.max(image)
    return im

@overall_runtime
def make_board_edge(image,board):
    im = copy.deepcopy(image)
    x,y,fb = board
    for t in range(round(fb*8)):
        for b in range(5):
            im[x-t,y+b] = np.max(image)
            im[x+b,y-t] = np.max(image)
            im[x-t,y+b-round(8*fb)] = np.max(image)
            im[x+b-round(8*fb),y-t] = np.max(image)
    return im


# Hier baue ich eine Clusterung der Boards nach fast identischer Lage
# So sollen die aufsummierten Scores robuster sein als die Einzelnen.
# Ja, sieht extrem robust aus.
@overall_runtime
def board_clustering(possible_boards, cutoff=4):
    clustered_boards = []
    for board1, score1 in possible_boards:
        score = 0.0
        count = 0
        x1, y1, fb1 = board1
        for board2, score2 in possible_boards:
            x2, y2, fb2 = board2
            if abs(x1 - x2) < cutoff and abs(y1 - y2) < cutoff and abs(fb1 - fb2) < 2.0:
                score += score2
                count += 1
        clustered_boards.append((board1, score, count))
    return sorted(clustered_boards, key=lambda item: item[1])[::-1]


# Und hier dann alles zusammenstecken:
@overall_runtime
def board_extraction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    norm_ie_img = normed_inner_edge_image(gray, size=8)

    # cutoff =  cutoff_berechnen(absolute(norm_ie_img), 2000)
    # punkte = punkte_aussuchen(absolute(norm_ie_img), cutoff)

    possible_boards = chessboard_voting(norm_ie_img)
    possible_boards = possible_boards[:1000]

    clustered_boards = board_clustering(possible_boards, cutoff=4)

    clustered_boards = [(b, p) for (b, p, n) in clustered_boards]

    non_overlapping_boards = remove_overlapping_boards(clustered_boards)

    return [(board, points) for (board, points) in non_overlapping_boards if points > 10000]

@overall_runtime
def largest_board_extraction(frame):
    boards = board_extraction(frame)
    if boards == []:
        return None
    else:
        boards = sorted([x for (x,y) in boards],key=lambda x:x[2])[::-1]
        board = boards[0]
    return board

# Und hier dann alles zusammenstecken:
@overall_runtime
def best_board_extraction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    norm_ie_img = normed_inner_edge_image(gray, size=8)

    print('kerneled')
    # cutoff =  cutoff_berechnen(absolute(norm_ie_img), 2000)
    # punkte = punkte_aussuchen(absolute(norm_ie_img), cutoff)

    possible_boards = chessboard_voting(norm_ie_img)
    print(len(possible_boards))
    possible_boards = possible_boards[:1000]

    clustered_boards = board_clustering(possible_boards, cutoff=4)

    return clustered_boards[0]


# Brett ausschneiden:
@overall_runtime
def schachbrett_auschneiden(img, board):
    x, y, fb = board
    return img[x - round(8 * fb):x, y - round(8 * fb):y]

@overall_runtime
def get_square(square, fb, gray_board):
    letters = 'ABCDEFGH'
    i1 = letters.index(square[0])
    i2 = int(square[1])

    x1 = round(fb * (8 - i2))
    x2 = round(fb * (8 - i2) + fb)

    y1 = round(fb * i1)
    y2 = round(fb * i1 + fb)

    return gray_board[x1:x2, y1:y2]
