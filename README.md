# Grounding meaning with associative learning using self-organizing maps

Python version and libraries:
- python (3.6)
- numpy (1.15.2)
- tensorflow (1.11.0)
- sklearn (0.20.0)
- matplotlib (3.0.0)

*How to run?*

- To start training, adjust parameters in main.py:prepare() and then run main.py. With current settings, code saves the model on every episode, which can then be used for plotting and reconstruction.
- To test vision reconstruction, optionally change word_to_meaning.py:path_to_models variable, and then run word_to_meaning.py
- Similary to vision, reconstruction, do the same for file meaning_to_word.py.
- To create graphs, run plotting.py or plotting_notebook.ipynb. Current settings save plots to disk.

*What is weird in code?*

- Code for plotting is not the most beautiful code I've ever written. Input reconstruction is also not very nice, but is manageable.
- Naming in other code differs slightly from naming used in report. Language vector is for example sometimes word_vector and sometimes string_vector. Comments in the code are consistent, however.

*Other notes*

- SOM code is taken mostly from https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
- Logging utility is as found on https://github.com/pat-coady/trpo/blob/master/src/utils.py
