


"""
neural_chess_bot_full_package.py

Massive all-in-one package (monolithic) that contains:
 A) PGN parsing (zip and folder), with per-move clock extraction
 B) Stockfish analysis integration (centipawn loss labeling)
 C) Opening-book builder and JSON writer
 D) Time-profile extractor (per-ply mean/std)
 E) Move vocabulary builder and TFRecords writer/reader
 F) TensorFlow model definitions: small and big (policy+value)
 G) Training pipelines (arrays and TFRecords), Colab-friendly
 H) Export helpers (SavedModel, HDF5, TFLite with quantization)
 I) Hybrid Lichess bot runner (berserk) with opening book + model + SF fallback
 J) Deployment helpers: systemd service unit example generator, Dockerfile template
 K) Colab notebook generator (writes .ipynb skeleton)

USAGE (examples):
  # Build CSV dataset from zip
  python neural_chess_bot_full_package.py --mode build_dataset --zip pgns.zip --out dataset.csv

  # Analyze dataset with Stockfish
  python neural_chess_bot_full_package.py --mode analyze_sf --csv dataset.csv --stockfish /usr/bin/stockfish --out dataset_sf.csv

  # Build opening book and time profile
  python neural_chess_bot_full_package.py --mode build_book_time --csv dataset_sf.csv --book_out opening_book.json --time_out time_profile.json

  # Train model on Colab/local
  python neural_chess_bot_full_package.py --mode train --csv dataset_sf.csv --model_out my_model --epochs 16 --batch_size 128

  # Export to TFLite for Raspberry Pi
  python neural_chess_bot_full_package.py --mode export_tflite --model_dir my_model --tflite_out model_pi.tflite --quantize

  # Run bot (hybrid berserk):
  python neural_chess_bot_full_package.py --mode run_bot --token <LICHESS_TOKEN> --model_dir my_model --book opening_book.json --time_profile time_profile.json --stockfish /usr/bin/stockfish


NOTE: This is a single-file monolith intended for developers who want everything in one place.
      For production, split into modules.

"""

# Standard libs
import os
import io
import sys
import re
import json
import time
import math
import random
import zipfile
import argparse
import logging
import tempfile
import threading
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Any, Tuple

# Third-party libs
import numpy as np
import pandas as pd
from tqdm import tqdm

import chess
import chess.pgn
import chess.engine

# TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# Optional lichess client
try:
    import berserk
    BERSERK_AVAILABLE = True
except Exception:
    BERSERK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('neural_chess_bot')

# Constants & defaults
PROMOTION_ORDER = ['q', 'r', 'b', 'n']
PROMO_BASE = 64 * 64
DEFAULT_INPUT_PLANES = 18
DEFAULT_SF_DEPTH = 16
DEFAULT_CHANNELS = 128
DEFAULT_BLOCKS = 8
DEFAULT_BATCH = 128
DEFAULT_EPOCHS = 8
DEFAULT_MOVE_VOCAB_SIZE = 4672

# Regex for PGN clock comments
PGN_CLK_RE = re.compile(r"\[%clk\s+([0-9:.]+)\]")

# ----------------------------- Utility functions -----------------------------

def uci_to_index(uci: str) -> Optional[int]:
    """Map a UCI move (e2e4 or e7e8q) to an integer index in a bounded space."""
    if not uci or len(uci) < 4:
        return None
    try:
        from_sq = chess.SQUARE_NAMES.index(uci[0:2])
        to_sq = chess.SQUARE_NAMES.index(uci[2:4])
    except ValueError:
        return None
    base = from_sq * 64 + to_sq
    if len(uci) == 5:
        prom = uci[4]
        if prom in PROMOTION_ORDER:
            return PROMO_BASE + (PROMOTION_ORDER.index(prom) * 4096) + base
        else:
            return None
    return base


def index_to_uci(idx: int) -> str:
    if idx < PROMO_BASE:
        from_sq = idx // 64
        to_sq = idx % 64
        return chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq]
    block = (idx - PROMO_BASE) // 4096
    base = (idx - PROMO_BASE) % 4096
    from_sq = base // 64
    to_sq = base % 64
    prom = PROMOTION_ORDER[block]
    return chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq] + prom


def clk_to_seconds(clk: Optional[str]) -> Optional[int]:
    if clk is None:
        return None
    try:
        parts = [int(p) for p in clk.split(':')]
    except Exception:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None

# ----------------------------- PGN Parsing -----------------------------

def parse_pgns_from_zip(zip_path: str, only_user: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse PGN files from a ZIP and return a list of move records.
    Each record: {game_id, white, black, result, fen, move_uci, clock, ply}
    """
    records = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = [n for n in z.namelist() if n.lower().endswith('.pgn')]
        for name in tqdm(files, desc='PGN files'):
            raw = z.read(name).decode('utf-8', errors='ignore')
            stream = io.StringIO(raw)
            while True:
                game = None
                try:
                    game = chess.pgn.read_game(stream)
                except Exception:
                    game = None
                if game is None:
                    break
                headers = dict(game.headers)
                if only_user:
                    if headers.get('White','').lower() != only_user.lower() and headers.get('Black','').lower() != only_user.lower():
                        continue
                node = game
                ply = 0
                while node.variations:
                    next_node = node.variations[0]
                    fen_before = node.board().fen()
                    mv_uci = next_node.move.uci()
                    comment = next_node.comment or ''
                    m = PGN_CLK_RE.search(comment)
                    clk = m.group(1) if m else None
                    records.append({'game_id': headers.get('GameId', headers.get('Site','') + '_' + headers.get('Date','') + '_' + str(ply)),
                                    'white': headers.get('White'), 'black': headers.get('Black'), 'result': headers.get('Result'),
                                    'fen': fen_before, 'move_uci': mv_uci, 'clock': clk, 'ply': ply})
                    node = next_node
                    ply += 1
    return records


def parse_pgn_folder(folder: str, only_user: Optional[str] = None) -> List[Dict[str, Any]]:
    records = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith('.pgn'):
                continue
            path = os.path.join(root, f)
            with open(path, 'r', errors='ignore') as fh:
                while True:
                    try:
                        game = chess.pgn.read_game(fh)
                    except Exception:
                        game = None
                    if game is None:
                        break
                    headers = dict(game.headers)
                    if only_user:
                        if headers.get('White','').lower() != only_user.lower() and headers.get('Black','').lower() != only_user.lower():
                            continue
                    node = game
                    ply = 0
                    while node.variations:
                        next_node = node.variations[0]
                        fen_before = node.board().fen()
                        mv_uci = next_node.move.uci()
                        comment = next_node.comment or ''
                        m = PGN_CLK_RE.search(comment)
                        clk = m.group(1) if m else None
                        records.append({'game_id': headers.get('GameId', headers.get('Site','') + '_' + headers.get('Date','') + '_' + str(ply)),
                                        'white': headers.get('White'), 'black': headers.get('Black'), 'result': headers.get('Result'),
                                        'fen': fen_before, 'move_uci': mv_uci, 'clock': clk, 'ply': ply})
                        node = next_node
                        ply += 1
    return records

# ----------------------------- Stockfish Integration -----------------------------

def analyze_with_stockfish(records: List[Dict[str, Any]], stockfish_path: str, depth: int = DEFAULT_SF_DEPTH, threads: int = 2) -> List[Dict[str, Any]]:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure({'Threads': threads})
    except Exception:
        pass
    results = []
    for r in tqdm(records, desc='SF analysis'):
        fen = r['fen']
        mv_uci = r['move_uci']
        board = chess.Board(fen)
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info.get('score')
            best_cp = score.pov(board.turn).score(mate_score=100000) if score else None
        except Exception:
            best_cp = None
        # after move
        try:
            mv = chess.Move.from_uci(mv_uci)
            board.push(mv)
            info2 = engine.analyse(board, chess.engine.Limit(depth=min(8, depth)))
            score2 = info2.get('score')
            mv_cp = score2.pov(not board.turn).score(mate_score=100000) if score2 else None
        except Exception:
            mv_cp = None
        cpl = None
        if best_cp is not None and mv_cp is not None:
            try:
                cpl = float(abs(best_cp - mv_cp))
            except Exception:
                cpl = None
        newr = dict(r)
        newr.update({'best_cp': best_cp, 'move_cp': mv_cp, 'centipawn_loss': cpl})
        results.append(newr)
    engine.quit()
    return results

# ----------------------------- Opening book & time profile -----------------------------

def build_opening_book(records: List[Dict[str, Any]], max_plies: int = 20, min_count: int = 1) -> Dict[str, Dict[str,int]]:
    book = defaultdict(Counter)
    grouped = defaultdict(list)
    for r in records:
        grouped[r['game_id']].append(r)
    for gid, moves in grouped.items():
        for r in moves[:max_plies]:
            book[r['fen']][r['move_uci']] += 1
    out = {fen: dict(cnts) for fen, cnts in book.items() if any(c >= min_count for c in cnts.values())}
    return out


def compute_time_profile(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    per_ply = defaultdict(list)
    for r in records:
        clk = r.get('clock')
        if not clk:
            continue
        secs = clk_to_seconds(clk)
        if secs is None:
            continue
        per_ply[str(r['ply'])].append(secs)
    profile = {}
    all_vals = []
    for ply, arr in per_ply.items():
        if len(arr) < 4:
            continue
        a = np.array(arr, dtype=np.float32)
        profile[ply] = {'mean': float(a.mean()), 'std': float(a.std()), 'count': int(a.size)}
        all_vals.extend(arr)
    if all_vals:
        a = np.array(all_vals, dtype=np.float32)
        profile['default'] = {'mean': float(a.mean()), 'std': float(a.std()), 'count': int(a.size)}
    else:
        profile['default'] = {'mean': 3.0, 'std': 1.5, 'count': 0}
    return profile

# ----------------------------- Move vocabulary & Dataset -----------------------------

def build_move_vocab(records: List[Dict[str, Any]], max_moves: int = DEFAULT_MOVE_VOCAB_SIZE) -> Tuple[Dict[str,int], Dict[int,str]]:
    cnt = Counter([r['move_uci'] for r in records])
    most = cnt.most_common(max_moves)
    move_to_id = {mv: i for i, (mv, _) in enumerate(most)}
    id_to_move = {i: mv for mv, i in move_to_id.items()}
    logger.info('Built move vocab size %d', len(move_to_id))
    return move_to_id, id_to_move


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    return df

# ----------------------------- Board -> Input Tensor -----------------------------

def board_to_planes(board: chess.Board, planes: int = DEFAULT_INPUT_PLANES) -> np.ndarray:
    arr = np.zeros((8,8,planes), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        row = 7 - (sq // 8)
        col = sq % 8
        pt = piece.piece_type - 1
        idx = pt + (0 if piece.color == chess.WHITE else 6)
        if idx < planes:
            arr[row, col, idx] = 1.0
    if planes >= 13:
        arr[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    if planes >= 17:
        arr[:, :, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        arr[:, :, 14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        arr[:, :, 15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        arr[:, :, 16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if planes >= 18:
        arr[:, :, 17] = min(board.fullmove_number / 200.0, 1.0)
    return arr

# ----------------------------- TFRecords -----------------------------

def write_tfrecords_from_records(records: List[Dict[str, Any]], move_to_id: Dict[str,int], out_prefix: str, shards: int = 8):
    n = len(records)
    per = max(1, n // shards)
    for s in range(shards):
        start = s * per
        end = min(n, (s+1)*per)
        path = f"{out_prefix}_{s:03d}.tfrecord"
        with tf.io.TFRecordWriter(path) as writer:
            for r in records[start:end]:
                mv = r['move_uci']
                if mv not in move_to_id:
                    continue
                midx = move_to_id[mv]
                board = board_to_planes(chess.Board(r['fen'])).astype(np.float32)
                bts = board.tobytes()
                cpl = float(r.get('centipawn_loss') or 0.0)
                clk = 0.0
                cs = r.get('clock')
                if cs:
                    ssecs = clk_to_seconds(cs)
                    clk = float(ssecs) if ssecs else 0.0
                feature = {
                    'board': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bts])),
                    'policy': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(midx)])),
                    'cpl': tf.train.Feature(float_list=tf.train.FloatList(value=[cpl])),
                    'clk': tf.train.Feature(float_list=tf.train.FloatList(value=[clk]))
                }
                ex = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(ex.SerializeToString())
        logger.info('Wrote TFRecord shard %s', path)


def parse_tfrecord_fn(example_proto):
    features = {
        'board': tf.io.FixedLenFeature([], tf.string),
        'policy': tf.io.FixedLenFeature([], tf.int64),
        'cpl': tf.io.FixedLenFeature([], tf.float32),
        'clk': tf.io.FixedLenFeature([], tf.float32)
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    board_raw = tf.io.decode_raw(parsed['board'], tf.float32)
    board = tf.reshape(board_raw, (8,8,DEFAULT_INPUT_PLANES))
    policy = tf.cast(parsed['policy'], tf.int32)
    aux = tf.stack([parsed['cpl'], parsed['clk']], axis=0)
    return (board, aux), policy

# ----------------------------- Model definitions -----------------------------

def residual_block(x, channels):
    y = layers.Conv2D(channels, 3, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(channels, 3, padding='same')(y)
    y = layers.BatchNormalization()(y)
    x = layers.add([x, y])
    x = layers.Activation('relu')(x)
    return x


def build_model(planes: int = DEFAULT_INPUT_PLANES, channels: int = DEFAULT_CHANNELS, blocks: int = DEFAULT_BLOCKS, move_vocab: int = DEFAULT_MOVE_VOCAB_SIZE) -> tf.keras.Model:
    inp = layers.Input(shape=(8,8,planes), name='board')
    x = layers.Conv2D(channels, 3, padding='same', activation='relu')(inp)
    for _ in range(blocks):
        x = residual_block(x, channels)
    # policy head
    p = layers.Conv2D(32, 1, activation='relu')(x)
    p = layers.Flatten()(p)
    p = layers.Dense(move_vocab, activation='softmax', name='policy')(p)
    # value head
    v = layers.Conv2D(8, 1, activation='relu')(x)
    v = layers.Flatten()(v)
    v = layers.Dense(256, activation='relu')(v)
    v = layers.Dense(1, activation='tanh', name='value')(v)
    model = tf.keras.Model(inputs=inp, outputs=[p, v])
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss={'policy': 'sparse_categorical_crossentropy', 'value': 'mse'}, metrics={'policy': 'accuracy'})
    logger.info('Built model with blocks=%d channels=%d move_vocab=%d', blocks, channels, move_vocab)
    return model

# ----------------------------- Training pipelines -----------------------------

def train_on_arrays(X: np.ndarray, Y: np.ndarray, AUX: np.ndarray, model_out: str, epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH):
    os.makedirs(model_out, exist_ok=True)
    n_moves = int(np.max(Y) + 1)
    model = build_model(move_vocab=n_moves)
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ckpt = callbacks.ModelCheckpoint(os.path.join(model_out, 'ckpt-{epoch:02d}.h5'), save_best_only=True)
    model.fit(ds, epochs=epochs, callbacks=[ckpt])
    model.save(os.path.join(model_out, 'saved_model'))
    logger.info('Training complete. Model saved to %s', model_out)
    return model


def train_on_tfrecords(tfrecord_paths: List[str], model_out: str, epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH):
    os.makedirs(model_out, exist_ok=True)
    ds = tf.data.TFRecordDataset(tfrecord_paths)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(20000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    ckpt = callbacks.ModelCheckpoint(os.path.join(model_out, 'ckpt-{epoch:02d}.h5'), save_best_only=True)
    model.fit(ds, epochs=epochs, callbacks=[ckpt])
    model.save(os.path.join(model_out, 'saved_model'))
    return model

# ----------------------------- Export helpers -----------------------------

def export_tflite(saved_model_dir: str, tflite_out: str, quantize: bool = False):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_out, 'wb') as f:
        f.write(tflite_model)
    logger.info('Exported TFLite to %s', tflite_out)
    return tflite_out

# ----------------------------- Bot runner (hybrid) -----------------------------

def load_opening_book(book_path: str) -> Dict[str, Dict[str,int]]:
    if not book_path or not os.path.exists(book_path):
        logger.info('No opening book found at %s', book_path)
        return {}
    with open(book_path, 'r') as f:
        data = json.load(f)
    # ensure Counter form
    out = {fen: Counter(moves) if isinstance(moves, list) else Counter(moves) for fen, moves in data.items()}
    logger.info('Loaded opening book with %d entries', len(out))
    return out


def load_time_profile(json_path: str) -> Dict[str, Any]:
    if not json_path or not os.path.exists(json_path):
        return {'default': {'mean': 3.0, 'std': 1.0, 'count': 0}}
    with open(json_path, 'r') as f:
        return json.load(f)


def sample_time(ply: int, profile: Dict[str, Any]) -> float:
    key = str(int(ply))
    if key in profile:
        p = profile[key]
        return max(0.05, random.gauss(p.get('mean', 1.0), p.get('std', 0.5)))
    d = profile.get('default', {'mean': 3.0, 'std': 1.0})
    return max(0.05, random.gauss(d['mean'], d['std']))


def choose_move_with_model(board: chess.Board, model: tf.keras.Model, temperature: float = 1.0) -> chess.Move:
    inp = board_to_planes(board).reshape(1,8,8,DEFAULT_INPUT_PLANES)
    out = model.predict(inp, verbose=0)
    if isinstance(out, (list, tuple)):
        probs = out[0][0]
    else:
        probs = out[0]
    legal = list(board.legal_moves)
    legal_moves = []
    legal_probs = []
    for m in legal:
        idx = uci_to_index(m.uci())
        if idx is None or idx >= len(probs):
            continue
        legal_moves.append(m)
        legal_probs.append(probs[idx])
    if not legal_moves:
        return random.choice(list(board.legal_moves))
    legal_probs = np.array(legal_probs)
    if temperature != 1.0:
        logits = np.log(np.clip(legal_probs, 1e-12, 1.0)) / temperature
        legal_probs = np.exp(logits - np.max(logits))
    legal_probs = legal_probs / (legal_probs.sum() + 1e-12)
    idx = np.random.choice(len(legal_moves), p=legal_probs)
    return legal_moves[idx]


def choose_move_with_stockfish(board: chess.Board, sf_path: str, depth: int = 12) -> chess.Move:
    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        res = engine.play(board, chess.engine.Limit(depth=depth))
        engine.quit()
        return res.move
    except Exception as e:
        logger.exception('Stockfish selection failed: %s', e)
        return random.choice(list(board.legal_moves))


def run_lichess_bot(token: str, model_dir: str, book_path: Optional[str] = None, time_profile_path: Optional[str] = None, sf_path: Optional[str] = None, temperature: float = 1.0):
    if not BERSERK_AVAILABLE:
        raise RuntimeError('berserk not installed. pip install berserk')
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    model = tf.keras.models.load_model(model_dir)
    book = load_opening_book(book_path) if book_path else {}
    time_profile = load_time_profile(time_profile_path) if time_profile_path else {'default': {'mean': 3.0, 'std': 1.0}}

    stream = client.bots.stream_incoming_events()
    logger.info('Listening for events...')
    for event in stream:
        typ = event.get('type')
        if typ == 'challenge':
            cid = event.get('id')
            try:
                client.bots.accept_challenge(cid)
                logger.info('Accepted challenge %s', cid)
            except Exception:
                logger.exception('Failed to accept challenge')
        elif typ == 'gameStart':
            gid = event['game']['id']
            threading.Thread(target=handle_game, args=(client, gid, model, book, time_profile, sf_path, temperature)).start()


def handle_game(client, gid: str, model: tf.keras.Model, book: Dict[str, Counter], time_profile: Dict[str, Any], sf_path: Optional[str], temperature: float):
    with client.bots.stream_game_state(gid) as gs:
        board = chess.Board()
        for state in gs:
            try:
                if state.get('status') in ('mate','resign','timeout','draw','stalemate'):
                    logger.info('Game ended: %s', gid)
                    break
                if state.get('isMyTurn'):
                    moves_text = state.get('moves', '')
                    board = chess.Board()
                    for mv in moves_text.split():
                        try:
                            board.push(chess.Move.from_uci(mv))
                        except Exception:
                            pass
                    fen = board.fen()
                    mv = None
                    if fen in book:
                        cnt = book[fen]
                        moves, weights = zip(*cnt.items())
                        total = sum(weights)
                        probs = [w/total for w in weights]
                        chosen = random.choices(moves, weights=probs, k=1)[0]
                        try:
                            m = chess.Move.from_uci(chosen)
                            if m in board.legal_moves:
                                mv = m
                                logger.info('Played book move %s', chosen)
                        except Exception:
                            mv = None
                    if mv is None:
                        ply = board.fullmove_number
                        think = sample_time(ply, time_profile)
                        time.sleep(min(think, 3.0))
                        try:
                            mv = choose_move_with_model(board, model, temperature)
                        except Exception:
                            logger.exception('Model failed to pick move')
                            mv = None
                    if mv is None and sf_path:
                        mv = choose_move_with_stockfish(board, sf_path)
                    if mv is None:
                        mv = random.choice(list(board.legal_moves))
                    try:
                        client.bots.make_move(gid, mv.uci())
                        logger.info('Game %s: played %s', gid, mv.uci())
                    except Exception:
                        logger.exception('Failed to send move')
            except Exception:
                logger.exception('Error in game loop')

# ----------------------------- Deployment Helpers -----------------------------

def generate_systemd_service(app_path: str, service_name: str = 'lichessbot', user: str = 'pi') -> str:
    unit = f"""[Unit]
Description=Lichess Bot - {service_name}
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={app_path}
ExecStart=/usr/bin/python3 {os.path.join(app_path, os.path.basename(__file__))} --mode run_bot --model_dir model --token $LICHESS_TOKEN
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    path = os.path.join(app_path, f'{service_name}.service')
    with open(path, 'w') as f:
        f.write(unit)
    logger.info('Wrote systemd service to %s', path)
    return path


def generate_dockerfile(app_dir: str, base_image: str = 'python:3.10-slim') -> str:
    dockerfile = f"""FROM {base_image}
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y wget git build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir python-chess tensorflow berserk stockfish numpy pandas tqdm
ENV LICHESS_TOKEN=""
CMD ["python", "{os.path.basename(__file__)}", "--mode", "run_bot", "--model_dir", "model"]
"""
    path = os.path.join(app_dir, 'Dockerfile')
    with open(path, 'w') as f:
        f.write(dockerfile)
    logger.info('Wrote Dockerfile to %s', path)
    return path

# ----------------------------- Colab Notebook Generator -----------------------------

def generate_colab_notebook(output_ipynb: str, pgn_zip_name: str = 'pgns.zip', stockfish_url: Optional[str] = None) -> str:
    nb = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 0}
    def add_cell(code: str):
        nb['cells'].append({"cell_type": "code", "metadata": {}, "source": code.splitlines(True), "outputs": []})
    add_cell("""# Upload your pgns.zip file\nfrom google.colab import files\nuploaded = files.upload()\n""")
    add_cell("""# Install requirements\n!pip install python-chess stockfish tensorflow berserk numpy pandas tqdm\n""")
    add_cell(f"""# Extract\nimport zipfile\nwith zipfile.ZipFile('{pgn_zip_name}','r') as z:\n    z.extractall('pgns')\nprint('extracted')\n""")
    add_cell("""# Then run the full pipeline script (upload it or paste)\n!python neural_chess_bot_full_package.py --mode build_dataset --zip pgns.zip --out dataset.csv\n""")
    with open(output_ipynb, 'w') as f:
        json.dump(nb, f)
    logger.info('Wrote Colab notebook to %s', output_ipynb)
    return output_ipynb

# ----------------------------- CLI Entrypoint -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, required=True, help='mode: build_dataset, analyze_sf, build_book_time, build_vocab, write_tfrecords, train_arrays, train_tfrecords, export_tflite, run_bot, gen_service, gen_docker, gen_colab')
    ap.add_argument('--zip', type=str, help='zip file with pgns')
    ap.add_argument('--folder', type=str, help='folder with pgns')
    ap.add_argument('--csv', type=str, help='csv dataset input')
    ap.add_argument('--out', type=str, help='output path')
    ap.add_argument('--stockfish', type=str, help='stockfish path')
    ap.add_argument('--model_out', type=str, help='model output dir')
    ap.add_argument('--model_dir', type=str, help='model dir for inference')
    ap.add_argument('--book', type=str, help='opening book json')
    ap.add_argument('--time_profile', type=str, help='time profile json')
    ap.add_argument('--token', type=str, help='lichess bot token')
    ap.add_argument('--tfrecord_prefix', type=str, help='tfrecord prefix')
    ap.add_argument('--tfrecords', type=str, help='comma-separated tfrecord paths')
    ap.add_argument('--tflite_out', type=str, help='tflite output path')
    args = ap.parse_args()

    mode = args.mode
    try:
        if mode == 'build_dataset':
            if not args.zip and not args.folder:
                raise SystemExit('provide --zip or --folder')
            if args.zip:
                recs = parse_pgns_from_zip(args.zip)
            else:
                recs = parse_pgn_folder(args.folder)
            df = records_to_dataframe(recs)
            out = args.out or 'dataset.csv'
            df.to_csv(out, index=False)
            logger.info('Wrote dataset %s (%d rows)', out, len(df))
        elif mode == 'analyze_sf':
            if not args.csv or not args.stockfish:
                raise SystemExit('--csv and --stockfish required')
            df = pd.read_csv(args.csv).to_dict('records')
            out = analyze_with_stockfish(df, args.stockfish)
            out_csv = args.out or 'dataset_sf.csv'
            pd.DataFrame(out).to_csv(out_csv, index=False)
            logger.info('Wrote analyzed CSV %s', out_csv)
        elif mode == 'build_book_time':
            if not args.csv:
                raise SystemExit('--csv required')
            df = pd.read_csv(args.csv).to_dict('records')
            book = build_opening_book(df)
            book_out = args.book or (args.out + '_book.json' if args.out else 'opening_book.json')
            with open(book_out, 'w') as f:
                json.dump(book, f)
            tp = compute_time_profile(df)
            tp_out = args.time_profile or (args.out + '_time.json' if args.out else 'time_profile.json')
            with open(tp_out, 'w') as f:
                json.dump(tp, f)
            logger.info('Wrote book %s and time profile %s', book_out, tp_out)
        elif mode == 'build_vocab':
            if not args.csv:
                raise SystemExit('--csv required')
            df = pd.read_csv(args.csv).to_dict('records')
            mv, id2 = build_move_vocab(df)
            outp = args.out or 'move_vocab.json'
            with open(outp, 'w') as f:
                json.dump(mv, f)
            logger.info('Wrote move vocab to %s', outp)
        elif mode == 'write_tfrecords':
            if not args.csv or not args.tfrecord_prefix:
                raise SystemExit('--csv and --tfrecord_prefix required')
            df = pd.read_csv(args.csv).to_dict('records')
            mv, id2 = build_move_vocab(df)
            write_tfrecords_from_records(df, mv, args.tfrecord_prefix)
        elif mode == 'train_arrays' or mode == 'train_arrays_simple':
            if not args.csv or not args.model_out:
                raise SystemExit('--csv and --model_out required')
            df = pd.read_csv(args.csv).to_dict('records')
            mv, id2 = build_move_vocab(df)
            X, Y, AUX = dataset_to_arrays_wrapper(df, mv)
            train_on_arrays(X, Y, AUX, args.model_out)
        elif mode == 'train_tfrecords':
            if not args.tfrecords or not args.model_out:
                raise SystemExit('--tfrecords and --model_out required')
            paths = args.tfrecords.split(',')
            train_on_tfrecords(paths, args.model_out)
        elif mode == 'export_tflite':
            if not args.model_dir or not args.tflite_out:
                raise SystemExit('--model_dir and --tflite_out required')
            export_tflite(args.model_dir, args.tflite_out, quantize=False)
        elif mode == 'run_bot':
            if not args.token or not args.model_dir:
                raise SystemExit('--token and --model_dir required')
            run_lichess_bot(args.token, args.model_dir, book_path=args.book, time_profile_path=args.time_profile, sf_path=args.stockfish)
        elif mode == 'gen_service':
            app = args.out or os.getcwd()
            generate_systemd_service(app)
        elif mode == 'gen_docker':
            app = args.out or os.getcwd()
            generate_dockerfile(app)
        elif mode == 'gen_colab':
            outp = args.out or 'colab_notebook.ipynb'
            generate_colab_notebook(outp, pgn_zip_name=(args.zip or 'pgns.zip'))
        else:
            raise SystemExit('Unknown mode')
    except Exception:
        logger.exception('Error in main:')

# Small wrapper helpers used in CLI training (kept near end for readability)

def dataset_to_arrays_wrapper(records: List[Dict[str, Any]], move_to_id: Dict[str,int]):
    X = []
    Y = []
    AUX = []
    missing = 0
    for r in tqdm(records, desc='records->arrays'):
        mv = r['move_uci']
        if mv not in move_to_id:
            missing += 1
            continue
        X.append(board_to_planes(chess.Board(r['fen'])))
        Y.append(move_to_id[mv])
        cpl = float(r.get('centipawn_loss') or 0.0)
        clk = 0.0
        cs = r.get('clock')
        if cs:
            ssec = clk_to_seconds(cs)
            clk = float(ssec) if ssec else 0.0
        AUX.append([cpl, clk])
    logger.info('Skipped %d records missing from vocab', missing)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32), np.array(AUX, dtype=np.float32)

if __name__ == '__main__':
    main()
