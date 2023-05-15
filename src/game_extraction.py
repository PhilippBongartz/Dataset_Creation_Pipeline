from PIL import Image
from src.board_extraction import *
from src.utils import *
from src.video_utils import *
import numpy as np
import math
import chess
import cv2
import copy
import pickle
import pandas as pd
import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras import models
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


#os.environ["OPENCV_LOG_LEVEL"]="SILENT"
import cv2
import numpy as np
import matplotlib.pyplot as plt


@overall_runtime
def all_legal_moves_from_full_fen(fen):
    board = chess.Board(fen)
    return [str(move) for move in board.legal_moves]

@overall_runtime
def all_possibly_legal_moves_from_rump_fen(fen):
    fen = fen.split()[0]
    white_moves = []
    for en_passant in ['a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6']:
        possible_fen = fen + ' w KQkq ' + en_passant + ' 0 1'
        white_moves += all_legal_moves_from_full_fen(possible_fen)
    white_moves = list(set(white_moves))

    black_moves = []
    for en_passant in ['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3']:
        possible_fen = fen + ' b KQkq ' + en_passant + ' 0 1'
        black_moves += all_legal_moves_from_full_fen(possible_fen)
    black_moves = list(set(black_moves))

    return white_moves + black_moves

@overall_runtime
def match2square(match, fb):
    x, y = match
    c = 'ABCDEFGH'[sorted([(abs(x - t * fb), t) for t in range(8)])[0][1]]
    c_err = sorted([(abs(x - t * fb), t) for t in range(8)])[0][0]
    r = '87654321'[sorted([(abs(y - t * fb), t) for t in range(8)])[0][1]]
    r_err = sorted([(abs(y - t * fb), t) for t in range(8)])[0][0]
    return c + r, c_err + r_err

@overall_runtime
def get_template_matches(template, board, threshold=0.8):
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(board, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return list(zip(*loc[::-1]))

@overall_runtime
def difference_extraction(brett1, brett2, fb):

    gray_board1 = cv2.cvtColor(brett1, cv2.COLOR_BGR2GRAY)
    gray_board2 = cv2.cvtColor(brett2, cv2.COLOR_BGR2GRAY)

    squares = []
    for c in 'ABCDEFGH':
        for r in '12345678':
            squares.append(c + r)
    differences = []
    for square in squares:
        template = get_square(square, fb, gray_board1)
        matches = get_template_matches(template, gray_board2, threshold=0.8)
        squares_in_matches = [match2square(match, fb) for match in matches]
        squares_in_matches = list(set([s for (s, n) in squares_in_matches]))
        if square not in squares_in_matches:
            # print(square)
            # print(squares_in_matches)
            # print()
            differences.append(square)
    return differences

@overall_runtime
def turn_the_board(prob_pos):
    return prob_pos[::-1]
    
    

@overall_runtime
def extract_bretter(video_path, n=10):
    """
    A generator that yield board pictures + frame indices
    extracted from a video.
    """
    cap = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    find_board_n = total_frames//10 # wir schauen nur 10 frames an um das board zu finden
    
    board = None
    for frame in read_every_nth_frame(video_path, find_board_n):
        if board is None:
            board = largest_board_extraction(frame)
        if board is not None:
            break
            
    if board is None:
        print('No board found!!!')
        return
    
    i=0
    for frame in read_every_nth_frame(video_path, n):
        brett = schachbrett_auschneiden(frame,board)
        yield (brett,i)
        i += n

@overall_runtime   
def format_square_image(im):
    """Formats the square images for the model"""
    resized = cv2.resize(im, (64,64), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    reshaped = gray.reshape((64,64,1))
    
    return reshaped

# The square classification model we use
piece_classifcation_tf_model = tf.keras.models.load_model('piece_color_classification_09994')

# turn pictures of boards into probabilistic positions - there is now a CLIP version and this...
@overall_runtime
def probabilistic_position(brett):
    """
    Takes the picture of a chess board and predicts the piece for each square.
    """

    fb = brett.shape[0] / 8
    # Order like in FEN starting with A8 row by row
    
    squares = []
    for r in '87654321':
        for c in 'ABCDEFGH':
            square = get_square(c + r, fb, brett)
            formatted_square = format_square_image(square)
            squares.append(formatted_square)

    inputs = np.array(squares)
    probabilistic_position = piece_classifcation_tf_model.predict(inputs,verbose=0)
    
    return probabilistic_position

# turn a probabilistic position into a fen
label2fensquare = ['B',1,'K','N','P','Q','R','b','k','n','p','q','r']

@overall_runtime
def max_likelihood_fen(probabilistic_position):
    """Extracts the likeliest fen from a prob_pos"""
    """rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"""
    position = []
    for p in probabilistic_position:
        piece = label2fensquare[np.argmax(p)]
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

# turn the board for black
@overall_runtime
def turn_the_board(prob_pos):
    return prob_pos[::-1]
    
    


@overall_runtime
def extract_unique_positions(video_path,n):
    frame_numbers  = []
    prob_positions = []
    fens           = []

    for (brett,frame_num) in extract_bretter(video_path,n=n):
        prob_pos = probabilistic_position(brett)
        fen      = max_likelihood_fen(prob_pos)
        if len(fens)==0 or fens[-1]!=fen:
            frame_numbers.append(frame_num)
            prob_positions.append(prob_pos)
            fens.append(fen)

    return frame_numbers, prob_positions, fens
    

@overall_runtime
def split_by_games(prob_positions,verbose=0):
    """
    Splits the prob_positions into games by looking at the number of pieces on starting squares.
    Start and end is the index of prob_positions not the frame number.
    """
    poss = [prob_pos2_pieces_on_start_squares(prob_pos) for prob_pos in prob_positions]
    poss_diffs = [x-z for (x,z) in zip(poss[1:],poss[:-1])]
    copl = [prob_pos2_color_played(prob_pos) for prob_pos in prob_positions]
    
    if verbose:
        plt.plot([t for t in range(len(prob_positions))],poss)
        plt.show()

    starts_and_ends = []
    looking_for_start = True
    for i in range(len(poss_diffs)):
        if looking_for_start and poss[i]>30.0:
            starts_and_ends.append((i,'start'))
            looking_for_start = False
        if poss[i]<29.0:
            looking_for_start = True

        if poss_diffs[i]>5:
            starts_and_ends.append((i,'end'))


    game_splits = []
    for i in range(len(starts_and_ends)-1):
        if starts_and_ends[i][1] == 'start' and starts_and_ends[i+1][1] == 'end':
            start = starts_and_ends[i][0]
            end   = starts_and_ends[i+1][0]

            if copl[start:end+1].count('white') > (end-start)/2:
                color = 'white'
            else:
                color = 'black'
            game_splits.append((start,end,color))
            
    return game_splits

@overall_runtime
def prob_pos2_color_played(prob_pos):
    white_piece_indices = [0,2,3,4,5,6]
    black_piece_indices = [7,8,9,10,11,12]

    balance  = prob_pos[0:16,white_piece_indices].sum() # FEN starts with row 8
    balance += prob_pos[48:,black_piece_indices].sum()
    balance -= prob_pos[0:16,black_piece_indices].sum() 
    balance -= prob_pos[48:,white_piece_indices].sum()
    
    if balance<0:
        return 'white'
    return 'black'

@overall_runtime
def prob_pos2num_of_pieces(prob_pos):
    return 64 - prob_pos[:,1].sum()

@overall_runtime
def prob_pos2_pieces_on_start_squares(prob_pos):
    return 32 - (prob_pos[:16,1].sum() + prob_pos[48:,1].sum())

@overall_runtime
def make_games(game_splits,frame_numbers,prob_positions,fens):
    """splitting prob_positions by games, turning black positions + recomputing fens"""
    games = []
    for start,end,color in game_splits:
        game_frame_numbers  = frame_numbers[start:end+1]
        game_prob_positions = prob_positions[start:end+1]
        game_fens           = fens[start:end+1]
        
        if color == 'black':
            game_prob_positions = [turn_the_board(prob_pos) for prob_pos in game_prob_positions]
            game_fens           = [max_likelihood_fen(prob_pos) for prob_pos in game_prob_positions]
        
        games.append([game_frame_numbers,game_prob_positions,game_fens])
    return games

@overall_runtime   
def fen_move2fen(fen,move):
    chess_board = chess.Board(fen)
    chess_board.push_san(move)
    return chess_board.fen()

@overall_runtime
def enpassant_possibilities(fen,color):
    
    fen = fen.split()[0]
    rows = fen.split('/')
    if color == 'white':
        enpassant_row = rows[3]
    else:
        enpassant_row = rows[4]
        
    for t in range(1,9):
        enpassant_row = enpassant_row.replace(str(t),'0'*t)

    if 'white':
        enpassant_row = enpassant_row.replace('Pp','Px') # only with opposite pawn next to pawn
        enpassant_row = enpassant_row.replace('pP','xP') # is enpassant possible
    else:
        enpassant_row = enpassant_row.replace('pP','Px') # only with opposite pawn next to pawn
        enpassant_row = enpassant_row.replace('Pp','xP') # is enpassant possible
        
    enpassants = ['-']
    for p,l in zip(enpassant_row,'abcdefgh'):
        if p == 'x':
            if color == 'white':
                enpassants.append(l+'6')    
            else:
                enpassants.append(l+'3')    

    return enpassants


@overall_runtime
def all_possibly_legal_moves_fens_from_rump_fen_by_color(fen):
    
    fen = fen.split()[0]
    
    white_moves_fens = []
    enpassants = enpassant_possibilities(fen,'white')
    for en_passant in enpassants:
        possible_fen = fen + ' w KQkq ' + en_passant + ' 0 1'
        white_moves  = all_legal_moves_from_full_fen(possible_fen)
        for move in white_moves:
            next_fen = fen_move2fen(possible_fen,move)
            white_moves_fens.append((move,next_fen.split()[0]))
    white_moves_fens = list(set(white_moves_fens))

    black_moves_fens = []
    enpassants = enpassant_possibilities(fen,'black')
    for en_passant in enpassants:
        possible_fen = fen + ' b KQkq ' + en_passant + ' 0 1'
        black_moves  = all_legal_moves_from_full_fen(possible_fen)
        for move in black_moves:
            next_fen = fen_move2fen(possible_fen,move)
            black_moves_fens.append((move,next_fen.split()[0]))
    black_moves_fens = list(set(black_moves_fens))

    return white_moves_fens , black_moves_fens

@overall_runtime
def connect_positions_with_one_move(prev,fen):    
    white_moves_fens , black_moves_fens = all_possibly_legal_moves_fens_from_rump_fen_by_color(prev)
        
    for move,nex in white_moves_fens:
        if nex == fen:
            connecting_move = move,'white'
            return connecting_move
                
    for move,nex in black_moves_fens:
        if nex == fen:
            connecting_move = move,'black'
            return connecting_move

@overall_runtime
def square_diffs_from_fens(fen1,fen2):
    fen1 = fen1.split()[0]
    fen2 = fen2.split()[0]
    
    fen1 = fen1.split('/')
    fen2 = fen2.split('/')
    
    diffs = []
    
    for row in range(8):
        if fen1[row] != fen2[row]:
            for t in range(1,9):
                fen1[row] = fen1[row].replace(str(t),'0'*t)
                fen2[row] = fen2[row].replace(str(t),'0'*t)
                
            for t in range(8):
                if fen1[row][t] != fen2[row][t]:
                    diffs.append('abcdefgh'[t]+'87654321'[row])
    return diffs

@overall_runtime
def restrict_moves_fens_to_diffs(moves,diffs):
    return [(move,fen) for (move,fen) in moves if (move[:2] in diffs or move[2:] in diffs)]

@overall_runtime
def fen_tree_search(fen1, fen2, diffs, depth, variation, color):
    """Rekursiv durch die Züge bis ply = depth gehen.
    Zurückgegeben wird die gefundene Variante zu fen2"""
    
    if depth == 0:
        return None
    
    white_moves_fens , black_moves_fens = all_possibly_legal_moves_fens_from_rump_fen_by_color(fen1)
    
    if color=='white':
        restricted_moves_fens = restrict_moves_fens_to_diffs(white_moves_fens, diffs)
    else:
        restricted_moves_fens = restrict_moves_fens_to_diffs(black_moves_fens, diffs)
        

    next_color = {'white':'black','black':'white'}

    result = None
    for move,next_fen in restricted_moves_fens:

        if next_fen == fen2:
            return variation+[move]
        
        next_diffs = square_diffs_from_fens(next_fen,fen2)
        
        result = fen_tree_search(next_fen, fen2, next_diffs, depth-1, variation+[move], next_color[color])
        
        if result is not None:
            return result
        
    return result

@overall_runtime    
def only_move_connected_fens(game,lookahead=4,depth=2,verbose=0):
    """
    Here we try to find those fens that are connected by one move.
    To this end we also skip fens.
    game = game_frame_numbers,game_prob_positions,game_fens
    """
    
    game_fens = game[2]
    game_frames = game[0]
    connected_fen_pairs = []
    frame_nums_of_pairs = []
    for i,fen in enumerate(game_fens):
        t = 1
        while t<lookahead and i+t<len(game_fens): 
            move = connect_positions_with_one_move(fen,game_fens[i+t])
            if move is not None:
                connected_fen_pairs.append((fen,move,game_fens[i+t]))
                frame_nums_of_pairs.append((game_frames[i],game_frames[i+t]))
                
                if verbose:
                    chess_board = chess.Board(fen+" b KQkq - 0 4")
                    display(chess_board)
                    print(move,t)
                    chess_board = chess.Board(game_fens[i+t]+" b KQkq - 0 4")
                    display(chess_board)
                    print('#'*15)

            
            diffs = square_diffs_from_fens(fen,game_fens[i+t])
            moves_white = fen_tree_search(fen,game_fens[i+t], diffs, depth, [], 'white')
            moves_black = fen_tree_search(fen,game_fens[i+t], diffs, depth, [], 'black')
            
            if moves_white is not None:
                connected_fen_pairs.append((fen,(moves_white,'white'),game_fens[i+t]))
                frame_nums_of_pairs.append((game_frames[i],game_frames[i+t]))
                
                if verbose:
                    chess_board = chess.Board(fen+" b KQkq - 0 4")
                    display(chess_board)
                    print(moves_white,t,'white')
                    chess_board = chess.Board(game_fens[i+t]+" b KQkq - 0 4")
                    display(chess_board)
                    print('#'*15)
                
            if moves_black is not None:
                connected_fen_pairs.append((fen,(moves_black,'black'),game_fens[i+t]))
                frame_nums_of_pairs.append((game_frames[i],game_frames[i+t]))
                
                if verbose:
                    chess_board = chess.Board(fen+" b KQkq - 0 4")
                    display(chess_board)
                    print(moves_black,t,'black')
                    chess_board = chess.Board(game_fens[i+t]+" b KQkq - 0 4")
                    display(chess_board)
                    print('#'*15)
                
            t+=1
    return connected_fen_pairs, frame_nums_of_pairs
    
    
@overall_runtime
def build_graph(connected_fen_pairs,frame_nums_of_pairs):
    """
    Nodes: fen,color(ie whose move),frame_num
    Edges: moves(to the next position/node),fen,color,frame_num
    """
    graph = {}

    for i,connected_fen_pair in enumerate(connected_fen_pairs):

        fen1, moves, fen2 = connected_fen_pair # node edge node
        frame_num1, frame_num2 = frame_nums_of_pairs[i] # neccesary to make graph acyclic

        if type(moves[0])==str: # single moves gibt es immer auch als mehrfach moves
            continue
            moves = ([moves[0]],moves[1]) # sonst # turn single move into list of one move

        # enter fens as nodes and moves as edges
        color = moves[1]
        if (fen1,color,frame_num1) not in graph:
            graph[(fen1,color,frame_num1)] = []
        # compute color of fen2:
        if color=='white':
            color2 = ['white','black'][len(moves[0])%2]
        else:
            color2 = ['black','white'][len(moves[0])%2]
        # add edge
        graph[(fen1,color,frame_num1)].append((moves[0],fen2,color2,frame_num2))
    return graph
    
    
@overall_runtime
def graph_traversal(graph):
    length_to   = {}
    predecessor = {}
    variation   = {}

    # compute start nodes:
    end_nodes = []
    for node in graph:
        for edge in graph[node]:
            moves,fen2,color2,frame_num2 = edge
            node2 = (fen2,color2,frame_num2)
            end_nodes.append(node2)
    for node in graph:
        if node not in end_nodes:
            length_to[node]   = 0
            predecessor[node] = None
            variation[node]   = []

    for node in graph:
        fen,color,frame_num = node
        #print(fen)
        for edge in graph[node]:
            moves,fen2,color2,frame_num2 = edge
            #print(' '*4,moves)
            node2 = (fen2,color2,frame_num2)
            if node2 in graph:
                #print(' '*8,node2)
                if node2 in length_to:
                    if length_to[node2] < length_to[node] + len(moves):
                        length_to[node2] = length_to[node] + len(moves)
                        predecessor[node2] = node
                        variation[node2] = variation[node] + moves
                else:
                    length_to[node2] = length_to[node] + len(moves)
                    predecessor[node2] = node
                    variation[node2] = variation[node] + moves
                #print(' '*8,length_to[node2],variation[node2])
                
    return length_to, predecessor, variation
    
@overall_runtime
def extract_longest_variation(variation):
    max_var_length = 0
    max_length_node = None
    for node in variation:
        if len(variation[node])>max_var_length:
            max_var_length = len(variation[node])
            max_length_node = node
    #print(max_var_length)
    #print(variation[max_length_node])
    if max_length_node is None:
        return None,[]
    return max_length_node, variation[max_length_node]


@overall_runtime
def extract_all_predecessors(node, predecessor):
    """
    Carefull: This is not all positions, some nodes are connected by several moves
    """
    all_nodes = [node]
    while predecessor[all_nodes[0]] is not None:
        all_nodes = [predecessor[all_nodes[0]]] + all_nodes
    return all_nodes


@overall_runtime
def delete_nodes_from_graph(graph,nodes):
    """
    Only frame_number relevant
    """
    frame_nums_to_delete = [fn for (fen,c,fn) in nodes]
    nodes_to_delete = [node for node in graph if node[2] in frame_nums_to_delete]
    
    for node in nodes_to_delete:
        graph.pop(node,None)
    
    for node in graph:
        obsolete_edges = [edge for edge in graph[node] if edge[3] in frame_nums_to_delete]
        for edge in obsolete_edges:
            graph[node].remove(edge)
    return graph



@overall_runtime
def extract_variations_and_nodes(original_graph,min_length=4,verbose=0):
    
    graph = copy.deepcopy(original_graph)
    
    extracted_vars_and_nodes = []

    for t in range(100):

        length_to, predecessor, variation = graph_traversal(graph)

        max_length_node, variation[max_length_node] = extract_longest_variation(variation)

        if max_length_node is None:
            break

        all_nodes = extract_all_predecessors(max_length_node, predecessor)

        graph = delete_nodes_from_graph(graph,all_nodes)

        if len(variation[max_length_node])>=min_length:
            extracted_vars_and_nodes.append((variation[max_length_node],all_nodes))
        else:
            break
        
        if verbose:
            print(len(variation[max_length_node]),variation[max_length_node])
        
    return extracted_vars_and_nodes
    
    

@overall_runtime
def load_transcript(video_path):
    path_parts = video_path.split('/')
    path_parts[-1] = 'prompted_whisper_base.pickle'
    transcript_path = '/'.join(path_parts)

    transcript = pickle.load(open(transcript_path,'rb'))
    return transcript



@overall_runtime
def frame2time(frame_num, fps):
    """frame number should be correct now"""
    return frame_num/fps

@overall_runtime
def frames_num_to_comment(frame_num1, frame_num2, transcript, fps):
    start = frame2time(frame_num1,fps) 
    end   = frame2time(frame_num2,fps) 
    comment = get_comments_per_timeintervall(transcript['segments'],start,end)
    return comment


@overall_runtime
def add_comments(extracted_vars_and_nodes,transcript,graph,fps,game_num=None,verbose=0):
    data = []
    for var,nodes in extracted_vars_and_nodes:
        for i,(fen1,color1,frame_num1) in enumerate(nodes[:-1]):
            (fen2,color2,frame_num2) = nodes[i+1]
            comment = frames_num_to_comment(frame_num1, frame_num2, transcript, fps)
            if verbose:
                chess_board = chess.Board(fen1+" b KQkq - 0 4")
                display(chess_board)
                print(frame_num1, frame_num2,comment,graph[(fen1,color1,frame_num1)])
            # Extract moves:
            moves = None
            for (moves_edge,fen_edge,color_edge,frame_num_edge) in graph[(fen1,color1,frame_num1)]:
                if frame_num_edge == frame_num2:
                    moves = moves_edge
            data.append((game_num,fen1,color1,moves,frame_num1,frame_num2,comment))
    return data
    


@overall_runtime
def recursive_extraction(pool,extracted):
    """
    extracted should always be just one (frame_num,prob_pos,fen)
    the frame_num should be at the beginning or the end of pool
    the other end is extracted and the fens are compared
    if they are identical --> the start extracted is returned
    if they are different --> pool is split in the middle and recursive_extraction is called twice
    """

    if len(pool)==1:
        return [extracted]
    
    (frame_num,prob_pos,fen) = extracted
    
    new_index = np.where(pool[0][0]==frame_num,len(pool)-1,0)
    
    new_frame_num = pool[new_index][0]
    new_prob_pos  = probabilistic_position(pool[new_index][1])
    new_fen       = max_likelihood_fen(new_prob_pos)
    
    #sorted_extracted = sorted([extracted,(new_frame_num,new_prob_pos,new_fen)])
    sorted_extracted = [extracted,(new_frame_num,new_prob_pos,new_fen)]
    sorted_extracted.sort(key=lambda elem: elem[0])
    
    
    if new_fen == fen:
        return [sorted_extracted[0]]
    
    middle = len(pool)//2
    
    first_half  = recursive_extraction(pool[:middle],sorted_extracted[0])
    
    second_half = recursive_extraction(pool[middle:],sorted_extracted[1])   
                                                                      
    return sorted(first_half + second_half)
    
    
    
@overall_runtime
def extract_unique_positions_fast(video_path,n, N):
    """
    In this function we collect N frames and extract probabilistic positions from them 
    only if necessary, ie if the position has changed we look for earlier changes.
    """
    
    frame_numbers  = []
    prob_positions = []
    fens           = []
    
    pool           = []

    for (brett,frame_num) in extract_bretter(video_path,n=n):
        
        pool.append((frame_num,brett))
        
        if len(pool)==N: # we extract the necessary number of positions
            
            frame_num = pool[0][0]
            prob_pos  = probabilistic_position(pool[0][1])
            fen       = max_likelihood_fen(prob_pos)
            extracted = (frame_num,prob_pos,fen)
    
            extracted_positions = recursive_extraction(pool,extracted)
            for (frame_num,prob_pos,fen) in extracted_positions:
                if fens==[] or fen!=fens[-1]:
                    frame_numbers.append(frame_num)
                    prob_positions.append(prob_pos)
                    fens.append(fen)
            pool = []
 
    if len(pool): # we extract the necessary number of positions
        frame_num = pool[0][0]
        prob_pos  = probabilistic_position(pool[0][1])
        fen       = max_likelihood_fen(prob_pos)
        extracted = (frame_num,prob_pos,fen)
            
        extracted_positions = recursive_extraction(pool,extracted)
        for (frame_num,prob_pos,fen) in extracted_positions:
            if fen!=fens[-1]:
                frame_numbers.append(frame_num)
                prob_positions.append(prob_pos)
                fens.append(fen)

    return frame_numbers, prob_positions, fens

@overall_runtime
def extract_unique_positions_fast2(video_path,n, N):
    """
    In this function we collect N frames and extract probabilistic positions from them 
    only if necessary, ie if the position has changed we look for earlier changes.
    """
    
    frame_numbers  = []
    prob_positions = []
    fens           = []
    
    pool           = []

    for (brett,frame_num) in extract_bretter(video_path,n=n):
        if pool==[]: # we always keep the last extracted - here we compute the first
            prob_pos  = probabilistic_position(brett)
            fen       = max_likelihood_fen(prob_pos)
            extracted = (frame_num,prob_pos,fen)
            
        pool.append((frame_num,brett))
        
        if len(pool)==N: # we extract the necessary number of positions
    
            extracted_positions = recursive_extraction(pool,extracted)
        
            for (frame_num,prob_pos,fen) in extracted_positions:
                if fens==[] or fen!=fens[-1]:
                    frame_numbers.append(frame_num)
                    prob_positions.append(prob_pos)
                    fens.append(fen)
            
            # we keep the last extracted and that frame in pool
            pool = [(frame_numbers[-1],None)] # we don't need the brett
            extracted = (frame_numbers[-1],prob_positions[-1],fens[-1])
 
    if len(pool):
        extracted_positions = recursive_extraction(pool,extracted)
        for (frame_num,prob_pos,fen) in extracted_positions:
            if fen!=fens[-1]:
                frame_numbers.append(frame_num)
                prob_positions.append(prob_pos)
                fens.append(fen)

    return frame_numbers, prob_positions, fens


@overall_runtime
def save_data_df(df,video_path):
    path_parts = video_path.split('/')
    path_parts[-1] = 'fen_comment_data.csv'
    data_path = '/'.join(path_parts)
    df.to_csv(data_path)
    
    













    
