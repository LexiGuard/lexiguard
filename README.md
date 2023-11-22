# lexiguards
Detecting toxicity in texts


## Environment

In some cases it is necessary to install the Rust compiler for the transformers library.

```BASH
brew install rustup
rustup-init
```
Than press ```1``` for the standard installation.

Then we can go on to install hdf5:

```BASH
 brew install hdf5
```
With the system setup like that, we can go and create our environment and install tensorflow

```BASH
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
