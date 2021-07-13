[![Anaconda-Server Badge](https://anaconda.org/fkretschmer/bertax/badges/version.svg)](https://anaconda.org/fkretschmer/bertax)

# BERTax: Taxonomic Classification of DNA sequences

## Installation
### Conda
Install in new conda environment
```shell
conda create -n bertax -c fkretschmer bertax
```

Activate environment and install necessary pip-dependencies
```shell
conda activate bertax
pip install keras-bert==0.86.0
```

### Local pip-only installation

Clone the repository (Git LFS has to be enabled beforehand to be able to download the large model weights file)
```shell
git lfs install # if not already installed
git clone https://github.com/f-kretschmer/bertax.git
```

Then install with pip
```shell
pip install -e bertax
```

## Docker

Alternatively to installing, a docker container is also available, pull and run:
```shell
docker run -t --rm -v /path/to/input/files:/in fkre/bertax:latest /in/sequences.fa
```

The docker container can also be run with GPU-support, likely resulting in much faster predictions. For this, the `nvidia-container-toolkit` has to be [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), the `bertax` image has to be run with the flag `--gpus all`.

The image can be built locally (after cloning -- see above) with
```shell
docker build -t bertax bertax
```

## Usage

The script takes a (multi)fasta as input and outputs a list of predicted classes to the console:
```shell
bertax sequences.fasta
```

Options:
<table>
<thead><tr><th>parameter</th><th>explanation</th></tr></thead>
<tbody>
<tr><td><code>-o</code> <code>--output_file</code></td><td>write output to specified file (tab-separated format) instead of to the output stream (console)</td></tr>
<tr><td><code>--conf_matrix_file</code></td><td>output confidences for all classes of all ranks to JSON file</td></tr>
<tr><td><code>--sequence_split</code></td><td>how to handle sequences sequence longer than the maximum (window) size: split into equal chunks (<code>equal_chunks</code>, default) or use random sequence window (<code>window</code>)</td></tr>
<tr><td><code>-C</code> <code>--maximum_sequence_chunks</code></td><td>maximum number of chunks to use per (long) sequence</td></tr>
<tr><td><code>--running_window</code></td><td>if enabled, a running window approach is chosen to go over each sequence to make predictions</td></tr>
<tr><td><code>--running_window_stride</code></td><td>stride for running window (default: 1)</td></tr>
<tr><td><code>--custom_window_size</code></td><td>allows specifying a custom, smaller window size</td></tr>
<tr><td><code>--chunk_predictions</code></td><td>output predictions per chunk, otherwise (by default) chunk predictions are averaged</td></tr>
<tr><td><code>--output_ranks</code></td><td>specify which ranks to include in output (default: superkingdom phylum genus)</td></tr>
<tr><td><code>--no_confidence</code></td><td>if set, do not include confidence scores in output</td></tr>
<tr><td><code>--batch_size</code></td><td>batch size (i.e., how many sequence chunks to predict at the same time); can be lowered to decrease memory usage and increased for better performance (default: 32)</td></tr>
<tr><td><code>-t</code> <code>--nr_threads</code></td><td>set the number of threads used (default: determine automatically)</td></tr>
</tbody>
</table>

Note, that "unknown" is a special placeholder class for each prediction rank, meaning the sequence's taxonomy is predicted to be unlike any possible output class.

### Examples

Default mode, sequences longer than 1500 nt are split into equal chunks, one prediction (average) per sequence
```shell
bertax sequences.fa
```

Only use one random chunk per sequence (for sequences longer than 1500 nt)
```shell
bertax --sequence_split window sequences.fa
```

Only output the superkingdom

```shell
bertax sequences.fa --output_ranks superkingdom
```

Predict with a running window in 300 nt steps and output predictions for all chunks (no threshold for the number of chunks per sequence)

```shell
bertax -C -1 --running_window --running_window_stride 300 --chunk_predictions sequences.fa
```

## Visualization

It is possible to get a visualization of the underlying BERT model, based on
[bertviz](https://github.com/jessevig/bertviz) for a specific DNA sequence. For this,
additional dependencies have to be installed:

- torch
- transformers
- bertviz==1.0.0

An HTML file with interactive visualization can be created with:

```shell
bertax-visualize sequence.fa
```

As visualization is quite performance-intensive for big sequences, parameters can be set
to only visualize a specific part (`-a $start -n $size`). Both an attention-head view and
model-view are available, set with the parameter `--mode {head|model}`.
