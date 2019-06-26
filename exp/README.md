# exp

This folder should be the default folder for containing experiment runs (e.g.
the default experiment folder for `train.py` is `exp/debug`). Don't be afraid
to use nested folders to for groups of related runs!

I'm still debating whether these results should be checked into Git or not.
Checking into Git is a convenient way to synchronize experiment results across
multiple machines and try to tie software updates to resulting experiments.

However, I've found that

1. In general, experimentation moves so fast that results quickly become
   obsolete and need to be replaced;
2. Stochasticity results in a lot of files changing that don't need to
   really be changed;
3. You'll end up with a LOT of experiments, which results in massive amounts
   of tracked files and slow `git status`es;
4. Git commits that just update experiment results are tiresome.

I think the best course of action is to leave this folder gitignored by
default. When you're sure you have FINAL final results (e.g.  for camera ready
publications) you can force-add the relevant results and explain them in the
project README.

In the meantime, you can manually sync gitignored experiments across
workstations with `rsync`/`scp`, with e.g. the following command:

```bash
# Sync .json and .csv files but ignore (potentially large) model files ending in *.pth
rsync -zarv --include="*/" --include="*.json" --include="*.csv" --exclude="*" "remote:~/path/to/remote/exp/" "exp/"
```
