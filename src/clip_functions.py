import torch
from PIL import Image
import open_clip
from src.board_extraction import *
from src.utils import *
import numpy as np
import math
import chess
import cv2


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

@overall_runtime
def piece_classification(square):
    image = Image.fromarray(square)
    image = preprocess(image).unsqueeze(0)
    text = open_clip.tokenize(["a horse",
                               "a crown with spikes",
                               "an imperial crown with a little cross on top",
                               "a rook shaped like a tower with crenels",
                               "a bishop's hat with a little cross",
                               "a pawn with a round head",
                               "an empty square",
                               "noisy texture"])

    pieces = ['N', 'Q', 'K', 'R', 'B', 'P', 'E', 'E']

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    if text_probs[0][0] < 0.90:
        text_probs[0][0] = 0.0
    if text_probs[0][5] < 0.80:
        text_probs[0][5] = 0.00

    print(text_probs)
    return pieces[np.argmax(text_probs)]


@overall_runtime
def piece_probabilities(square):
    image = Image.fromarray(square)
    image = preprocess(image).unsqueeze(0)
    text = open_clip.tokenize(["a horse",
                               "a crown with spikes",
                               "an imperial crown with a little cross on top",
                               "a rook shaped like a tower with crenels",
                               "a bishop's hat with a little cross",
                               "a pawn with a round head",
                               "an empty square",
                               "noisy texture"])

    pieces = ['N', 'Q', 'K', 'R', 'B', 'P', 'E', 'E']

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    if text_probs[0][0] < 0.90:
        text_probs[0][0] = 0.0
    if text_probs[0][5] < 0.80:
        text_probs[0][5] = 0.00

    text_probs[0][6] = text_probs[0][6] + text_probs[0][7]
    text_probs[0][7] = 0.0
    return text_probs[0][:7]

@overall_runtime
def piece_classification_plus(square):
    image = Image.fromarray(square)
    image = preprocess(image).unsqueeze(0)
    text = open_clip.tokenize(["a horse",
                               "a crown with spikes",
                               "an imperial crown with a little cross on top",
                               "a rook shaped like a tower with crenels",
                               "a bishop's hat with a little cross",
                               "a pawn with a round head",
                               "an empty square",
                               "noisy texture"])

    pieces = ['N', 'Q', 'K', 'R', 'B', 'P', 'E', 'E']

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    if text_probs[0][0] < 0.90:
        text_probs[0][0] = 0.0
    if text_probs[0][5] < 0.80:
        text_probs[0][5] = 0.00
    text_probs[0][6] = text_probs[0][6] + text_probs[0][7]

    print(text_probs)
    return pieces[np.argmax(text_probs)] + '_' + str(int(float(text_probs[0][np.argmax(text_probs)]) * 100))

@overall_runtime
def empty_square_prediction(square):
    """
    Hier stecken wir noch ein stärkeres Modell rein:
    """
    image = Image.fromarray(square)
    image = preprocess(image).unsqueeze(0)

    text = open_clip.tokenize(["a horse",
                               "a crown with spikes",
                               "an imperial crown with a little cross on top",
                               "a rook shaped like a tower with crenels",
                               "a bishop's hat with a little cross",
                               "a pawn with a round head",
                               "an empty square",
                               "noisy wooden texture"])

    pieces = ['N', 'Q', 'K', 'R', 'B', 'P', 'E', 'E']

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return text_probs[0][6] + text_probs[0][7]


squares = []
cols = 'ABCDEFGH'
rows = '12345678'
for c in cols:
    for r in rows:
        square = c + r
        squares.append(square)

middle_squares = []
cols = 'ABCDEFGH'
rows = '3456'
for c in cols:
    for r in rows:
        square = c + r
        middle_squares.append(square)

upper_squares = []
cols = 'ABCDEFGH'
rows = '78'
for c in cols:
    for r in rows:
        square = c + r
        upper_squares.append(square)

lower_squares = []
cols = 'ABCDEFGH'
rows = '12'
for c in cols:
    for r in rows:
        square = c + r
        lower_squares.append(square)



@overall_runtime
def empty_middle_squares(fb, brett):
    """
    This function checks how many of the emptiest squares are in the middle.
    So it also checks whether the top and bottom rows are non-empty.
    It is more or less a starting position detector.
    """
    square_preds = []
    for square in squares:
        square_img = get_square(square, fb, brett)
        pred = empty_square_prediction(square_img)
        square_preds.append((pred, square))
    emptiest_square = [s for (e, s) in sorted(square_preds)[32:]]
    # print(emptiest_square)
    return len(set(emptiest_square).intersection(set(middle_squares)))


@overall_runtime
def count_occupied_squares(fb, brett):
    """
    This function checks how many of the squares are occupied by pieces.
    """
    square_preds = []
    for square in squares:
        square_img = get_square(square, fb, brett)
        pred = empty_square_prediction(square_img)
        square_preds.append((pred, square))
        
    occupied_squares = 64 - sum([e for (e, s) in square_preds])

    return float(occupied_squares)


@overall_runtime
def empty_squares(fb, brett):
    """
    Wie viele Figuren sind noch auf dem Brett?
    """
    square_preds = []
    for square in squares:
        square_img = get_square(square, fb, brett)
        pred = empty_square_prediction(square_img)
        square_preds.append(pred)
    return sum(square_preds)

@overall_runtime
def color_of_piece_classification(square):
    image = Image.fromarray(square)
    image = preprocess(image).unsqueeze(0)
    text = open_clip.tokenize(["a white chess piece",
                               "a black chess piece"])

    colors = ['White', 'Black']

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    #print(text_probs)
    return colors[np.argmax(text_probs)]

@overall_runtime
def color_of_piece_probability(square):
    image = Image.fromarray(square)
    image = preprocess(image).unsqueeze(0)
    text = open_clip.tokenize(["a white chess piece",
                               "a black chess piece"])

    colors = ['White', 'Black']

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    #print(text_probs)
    return text_probs

@overall_runtime
def probabilistic_position_old(brett):
    fb = brett.shape[0] / 8
    """Order like in FEN starting with A8 row by row"""
    prob_position = []

    for r in '87654321':
        for c in 'ABCDEFGH':
            square = get_square(c + r, fb, brett)
            # white black
            color_prob = color_of_piece_probability(square)
            # print(color_prob)
            # 'N', 'Q', 'K', 'R', 'B', 'P', 'E'
            piece_prob = piece_probabilities(square)
            prob_position.append((color_prob, piece_prob))

            # plotting(square,title=c+r)

    return prob_position

# Einmaliges Encoden der Textanteile für die CLIP-Classifier:
color_words = ["a white chess piece","a black chess piece"]
color_text = open_clip.tokenize(color_words)
color_text_features = model.encode_text(color_text )
color_text_features /= color_text_features.norm(dim=-1, keepdim=True)

piece_words = ["a horse",
                "a crown with spikes",
                "an imperial crown with a little cross on top",
                "a rook shaped like a tower with crenels",
                "a bishop's hat with a little cross",
                "a pawn with a round head",
                "an empty square",
                "noisy texture"]
piece_text = open_clip.tokenize(piece_words)
piece_text_features = model.encode_text(piece_text )
piece_text_features /= piece_text_features.norm(dim=-1, keepdim=True)

both_text = open_clip.tokenize(piece_words+color_words)
both_text_features = model.encode_text(both_text )
both_text_features /= both_text_features.norm(dim=-1, keepdim=True)


@overall_runtime
def probabilistic_position(brett):
    """Das ist die optimierte Version: """

    # Squares extraction
    fb = brett.shape[0] / 8
    squares = []
    for r in '87654321':
        for c in 'ABCDEFGH':
            square = get_square(c + r, fb, brett)
            squares.append(square)

    # square image encoding
    images = []
    for square in squares:
        image = Image.fromarray(square)
        image = preprocess(image).unsqueeze(0)
        images.append(image)
    image_batch = torch.cat(images, 0)

    # CLIP probabilities
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ both_text_features.T).softmax(dim=-1)

    # building the probabilistic position
    piece_probs = text_probs[:, :8]
    color_probs = text_probs[:, 8:]
    
    # postprocessing: 
    piece_probs = [piece_probs[t,:] for t in range(64)]
    for text_probs in piece_probs:
        #if text_probs[0] < 0.90:
        #    text_probs[0] = 0.0
        #if text_probs[5] < 0.80:
        #    text_probs[5] = 0.00

        text_probs[6] = text_probs[6] + text_probs[7]
        text_probs = text_probs[:7]
    
    # combining color probs and piece probs
    prob_position = [x for x in zip(color_probs, piece_probs)]

    return prob_position




@overall_runtime
def max_likelihood_position(probabilistic_position):
    """rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"""
    position = []
    for (cp, pp) in probabilistic_position:
        piece = ['N', 'Q', 'K', 'R', 'B', 'P', 1][np.argmax(pp)]
        # print(pp,np.argmax(pp),piece)
        if piece != 1:
            # print(np.argmax(cp))
            if np.argmax(cp) == 1:
                piece = piece.lower()
        position.append(piece)

    fen_like = []
    f = -1
    for t in range(8):
        for tt in range(8):
            f += 1
            if position[f] != 1:
                fen_like.append(position[f])
            else:
                if f > 0 and type(fen_like[-1]) is int:
                    fen_like[-1] += 1
                else:
                    fen_like.append(position[f])
        fen_like.append('/')
    return ''.join([str(p) for p in fen_like[:-1]])




@overall_runtime
def which_color_is_being_played(fb, brett):
    """
    Aka upper_vs_lower_squares_brightness
    This function checks whether the upper or the lower squares are brighter.
    In the starting position this tells us whether white or black is being played.
    TODO: Don't get_square for all separately, but get top part of board etc.
    """
    upper_mean = 0.0
    for square in upper_squares:
        square_img = get_square(square, fb, brett)
        upper_mean += np.mean(square_img)

    lower_mean = 0.0
    for square in lower_squares:
        square_img = get_square(square, fb, brett)
        lower_mean += np.mean(square_img)

    if lower_mean > upper_mean:
        return "white"
    return "black"



@overall_runtime
def FEN2prob(fen, probabilistic_pos, min_prob=0.001):
    """Die Logprobability für eine Stellung wird berechnet."""

    piece2index = {'N': 0, 'Q': 1, 'K': 2, 'R': 3,
                   'B': 4, 'P': 5, '1': 6, 'n': 0,
                   'q': 1, 'k': 2, 'r': 3, 'b': 4, 'p': 5}

    color2index = {'N': 0, 'Q': 0, 'K': 0, 'R': 0,
                   'B': 0, 'P': 0, 'n': 1,
                   'q': 1, 'k': 1, 'r': 1, 'b': 1, 'p': 1}

    for no in range(8, 1, -1):
        fen = fen.replace(str(no), '1' * no)
    fen = fen.replace('/', '')
    fen = fen[:64]

    log_prob = 0

    for i, b in enumerate(fen):
        (cp, pp) = probabilistic_pos[i]
        log_prob += math.log(pp[piece2index[b]] + min_prob)
        if b != '1':
            log_prob += math.log(cp[color2index[b]] + min_prob)
    return log_prob





