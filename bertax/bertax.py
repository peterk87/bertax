import json
import logging
import os
import sys
import warnings

from time import time_ns
from enum import Enum
from pathlib import Path
from random import sample
from sys import version_info
from typing import List, Optional


import numpy as np
import pkg_resources
import typer
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm
import tensorflow as tf

from bertax import __version__
import bertax.utils as utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
warnings.simplefilter(action="ignore", category=FutureWarning)

MAX_SIZE = 1500
FIELD_LEN_THR = 50  # prevent overlong IDs

app = typer.Typer()


def version_callback(value: bool):
    if value:
        typer.echo(f"BERTax version {__version__}")
        raise typer.Exit()


def check_max_len_arg(value: int) -> int:
    value = int(value)
    if value < 1 or value > MAX_SIZE:
        raise typer.BadParameter(f"value has to be between 1 and {MAX_SIZE}")
    return value


class SeqSplit(str, Enum):
    equal_chunks = 'equal_chunks'
    window = 'window'


@app.command(
    epilog=f'BERTax version {__version__}; Python {version_info.major}.{version_info.minor}.{version_info.micro}'
)
def main(fastx: Path,
         output_file: Path = typer.Option(None, '-o', help='BERTax output'),
         custom_model_file: Path = typer.Option(None, '-m', '--custom-model-file',
                                                help="Custom finetuned BERTax model to use"),
         nr_threads: int = typer.Option(None, '-t', '--nr-threads',
                                        help="set the number of threads used (Default: determine automatically)"),
         conf_matrix_file: Path = typer.Option(None,
                                               help="if set, writes confidences for all possible classes in all ranks to specified file (JSON)"),
         sequence_split: SeqSplit = typer.Option(SeqSplit.equal_chunks, help="how to handle sequences longer "
                                                                             "than the maximum (window) size: split to equal chunks or "
                                                                             "use random sequence window; also see `--chunk_predictions` "
                                                                             "and `--running_window` (Default: equal_chunks)"),
         chunk_predictions: bool = typer.Option(False, help="if set, predictions on chunks for long sequences "
                                                            "are not averaged (default) but printed individually in the output"),
         running_window: bool = typer.Option(False, help="if enabled, a running window "
                                                         "approach is chosen to go over each sequence with a default stride of `1` "
                                                         "and make predictions on those windows; takes priority over `sequence_split`"),
         running_window_stride: int = typer.Option(90, help="stride (nt) for the running window approach"),
         custom_window_size: int = typer.Option(MAX_SIZE,
                                                '-s',
                                                '--custom-window-size',
                                                callback=check_max_len_arg,
                                                help=f"set custom, smaller window size (in nt, preferably multiples of 3), if unset, the maximum size of {MAX_SIZE} is used"),
         maximum_sequence_chunks: int = typer.Option(100, '-C', '--maximum-sequence-chunks',
                                                     help="maximum number of chunks to use for long sequences, -1 to use all chunks"),
         output_ranks: List[str] = typer.Option(['superkingdom', 'phylum', 'genus'],
                                                help='Rank predictions to include in output'),
         batch_size: int = typer.Option(32, help='Batch size for predictions'),
         force: bool = typer.Option(False, help='Force overwrite output file.'),
         verbose: bool = typer.Option(False, '-v', '--verbose',
                                      help='more verbose output if set, including progress bars'),
         quiet: bool = typer.Option(False, help='Do not log anything'),
         version: Optional[bool] = typer.Option(
             None,
             callback=version_callback,
             help=f'Print "BERTax version {__version__}" and exit',
         ),
         ):
    """BERTax: Predicting sequence taxonomy
    """
    from rich.traceback import install

    install(show_locals=True, width=120, word_wrap=True)

    logging.basicConfig(
        format="%(message)s",
        datefmt="[%Y-%m-%d %X]",
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True, console=Console(stderr=True, quiet=quiet))],
    )
    if output_file is None:
        logging.info('Output file not specified. Writing output to STDOUT.')
        output_file = sys.stdout
    else:
        if output_file.exists() and not force:
            logging.error(f'Output file "{output_file}" already exists! Specify "--force" to overwrite output file.')
            sys.exit(1)
    if custom_model_file is None:
        model_file = pkg_resources.resource_filename(
            "bertax", "resources/big_trainingset_all_fix_classes_selection.h5"
        )
    else:
        model_file = custom_model_file
    if nr_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(nr_threads)
        tf.config.threading.set_intra_op_parallelism_threads(nr_threads)
    logging.info(f'Running BERTax version {__version__} with model "{model_file}": {locals()}')
    model = utils.load_bert(model_file)
    logging.warning(f'model={model} | type={type(model)}')

    max_seq_len = (
        MAX_SIZE if custom_window_size is None else custom_window_size
    )
    token_dict = utils.get_token_dict()
    # convert input to processable tokens
    out = []
    with utils.get_file_handle(fastx) as fh, open(output_file, "w") if isinstance(output_file, Path) else output_file as handle:
        handle.write(
            "\t".join(["id"] + [f"{rank}_label\t{rank}_confidence" for rank in output_ranks])
            + "\n"
        )
        for seq_id, seq in tqdm(utils.parse_fastx(fastx, fh), unit='seq'):
            start_time_ns = time_ns()
            no_chunks = not running_window and (len(seq) <= max_seq_len or sequence_split == "window")
            if no_chunks:
                inputs = [utils.seq2tokens(seq, token_dict, np.ceil(max_seq_len / 3).astype(int),
                                     max_length=model.input_shape[0][1], )]
            else:
                chunks, positions = utils.seq_frames(
                    seq, max_seq_len, running_window, running_window_stride
                )
                if 0 < maximum_sequence_chunks < len(chunks):
                    logging.debug(f"sampling {maximum_sequence_chunks} from {len(chunks)} chunks")
                    chunks, positions = list(zip(*sample(list(zip(chunks, positions)), maximum_sequence_chunks)))
                inputs = [
                    utils.seq2tokens(
                        chunk,
                        token_dict,
                        np.ceil(max_seq_len / 3).astype(int),
                        max_length=model.input_shape[0][1],
                    )
                    for chunk in chunks
                ]
            time_ns_seq2tokens = time_ns()
            logging.debug(f"converted sequence '{seq_id}' (length {len(seq)}) into {len(inputs)} chunks ({(time_ns_seq2tokens - start_time_ns) / 1000000} ms)")
            processed_inputs = utils.process_bert_tokens_batch(inputs)
            time_ns_processed = time_ns()
            logging.debug(f"predicting sequence chunks for '{seq_id}' ({(time_ns_processed - time_ns_seq2tokens) / 1000000} ms)")
            preds = model.predict(processed_inputs, batch_size=batch_size)
            logging.debug(f"Prediction in {(time_ns() - time_ns_processed) / 1000000} ms")
            if chunk_predictions and not no_chunks:
                for pos, pred in zip(positions, [[p[i] for p in preds] for i in range(len(positions))]):
                    out_id = f"{seq_id}|{pos[0]}..{pos[1]}"
                    annotated = utils.annotate_predictions(pred)
                    write_prediction_result(handle, out_id, annotated)
                    if conf_matrix_file:
                        out.append((out_id, annotated))
            else:  # for each window
                preds = list(map(lambda p: p.mean(axis=0), preds))
                annotated = utils.annotate_predictions(preds)
                write_prediction_result(handle, seq_id, annotated)
                if annotated:
                    out.append((seq_id, annotated))
            logging.debug(f'elapsed: {(time_ns() - start_time_ns) / 1000000} ms')
    logging.info(f"predicted classes for {len(out)} sequence records")
    # OUTPUT
    if conf_matrix_file is not None:
        with open(conf_matrix_file, "w") as handle:
            json.dump(out, handle, indent=2)


def write_prediction_result(handle, seq_id, annotated) -> None:
    handle.write("\t".join([seq_id] + [f"{label}\t{confidence:.4%}" for rank, label, confidence in
                                       utils.best_predictions(annotated)]) + "\n")


if __name__ == "__main__":
    app()  # pragma: no cover
