# pytorch-project-template

This is a template for PyTorch machine learning projects. It's built with the
following principles in mind:

- Reproducibility: save detailed experiment metrics as the experiment runs.
    Save *all arguments* provided to a specific run *along with a hash of the
    git commit used to generate the code*, so you know exactly which version of
    the code generated which result (provided you don't have untracked
    changes).
- Flexibility: train and load models on or offcluster, with or without CUDA, and
  easily synchronize results across different machines. Also built for
  compatibility with [CodaLab](http://worksheets.codalab.org/).
- Modularity: as much as possible, keep code for specifying models, processing
  data, and building optimizations/loss functions in separate modules so they
  can be tweaked and reused across training/testing/visualization scripts. I
  unnecessary duplication of code often occurs when (1) building a model for
  training vs (2) loading a model for post-hoc analysis, and (i) running a
  model's training loop vs (ii) running a model's testing loop.

This may not be the ideal way of structuring projects, but it's the one I've
settled upon after my own process of trial and error. I'm always open to
suggestions for how to improve this workflow!

# TODO

One thing this is clearly missing is integration with tensorboard/visdom rather
than manually saving metrics to a file. Even without such integration, this
repository lacks ways of tracking experiment metrics as the experiment
progresses (besides just viewing `metrics.json`). I tend to analyze metrics
files in `R` (because I'm crazy) but you will need to spin your own solution
for now.
