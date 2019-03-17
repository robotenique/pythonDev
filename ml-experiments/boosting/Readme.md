
# TCC

## Anaconda environment
This work uses the [Anaconda tool](https://www.anaconda.com), especially, it uses the `conda` tool for managing the environment configuration. You can install the current environment with:

```bash 
$ conda env create --file=env.yaml
```

And if any change is made, you can save it again with:

```bash 
$ conda env export >! env.yaml
```

## Jupyter notebook

Use this command in the terminal for better visualization in *Jupyter notebook* (needs `jupyterthemes`)

```bash
$ jt -t grade3 -f roboto -fs 100 -altp -tfs 11 -nfs 115 -cellw 95% -T
```
