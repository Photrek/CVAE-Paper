"""Tells the shell pipelines which steps are already finished, so a resumed Colab run skips them.

Uses only the standard library, so the check costs milliseconds. Importing torch in run.py costs seconds per call,
and run.py also scans 200k CelebA files before it can decide to skip. Exit code 0 means "done, skip it".

    python pipeline_state.py step    <dataset> <model> <param> <dim> <samples> <mode>
    python pipeline_state.py corrupt <dataset>
    python pipeline_state.py metrics <dataset> <model> <dim> <samples> [robustness]

The file names here must match the ones written by run.py and the metric scripts.
"""
import glob
import os
import re
import sys

ROOT = "/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper"
LOCAL_EVAL = "/content/evaluation_dataset"
MODEL_FOLDERS = {1: "cvae", 2: "heavy_vae", 3: "beta_vae", 4: "prior_vae"}
CORRUPTIONS = ["gaussian_noise", "motion_blur", "fog", "shot_noise"]
MNIST_TEST_SIZE = 5000  # run.py splits the 10k MNIST test set into 5k validation and 5k test


def read_run_py_constant(name):
    """Reads a hyperparameter from run.py, so this file cannot drift from it."""
    run_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
    with open(run_py) as f:
        return int(re.search(rf"^{name}\s*=\s*(\d+)", f.read(), re.M).group(1))


def data_name(dataset):
    return "celeba_data" if dataset == 1 else "mnist_data"


def eval_set_size(dataset):
    size = read_run_py_constant("evaluation_set_size")
    return size if dataset == 1 else min(size, MNIST_TEST_SIZE)


def param_str(model, param):
    return f"{'kappa' if model in (1, 2) else 'beta'}_{float(param)}"


def output_dir(dataset, model, param, dim, samples):
    return os.path.join(ROOT, data_name(dataset), MODEL_FOLDERS[model],
                        f"outputs_{param_str(model, param)}_dim_{dim}_samples_{samples}")


def count_pngs(folder):
    return len(glob.glob(os.path.join(folder, "*.png")))


def training_done(out_dir, model, param, dim):
    log_path = os.path.join(out_dir, "epoch_log.txt")
    ckpt_path = os.path.join(out_dir, f"{MODEL_FOLDERS[model]}_{param_str(model, param)}_dim_{dim}_latest.pth")
    if not (os.path.exists(log_path) and os.path.exists(ckpt_path)):
        return False
    with open(log_path) as f:
        rows = [line.split("\t") for line in f.read().splitlines()[1:] if line.strip()]
    if not rows:
        return False
    if rows[-1][1].strip() == "nan":
        return True  # the run hit a numerical crash and run.py exits at once on every retry
    if not rows[-1][0].strip().isdigit() or int(rows[-1][0]) < read_run_py_constant("number_of_epochs"):
        return False
    # The checkpoint is saved after the last log row. An older checkpoint means a crash in between.
    return os.path.getmtime(ckpt_path) >= os.path.getmtime(log_path)


def step_done(dataset, model, param, dim, samples, mode):
    out_dir = output_dir(dataset, model, param, dim, samples)
    tag = param_str(model, param)
    run_name = f"{MODEL_FOLDERS[model]}_{tag}_dim_{dim}_samples_{samples}"
    expected = eval_set_size(dataset)
    if mode == 1:
        return training_done(out_dir, model, param, dim)
    if mode == 2:
        return count_pngs(os.path.join(LOCAL_EVAL, f"reconstructions_{run_name}")) >= expected
    if mode == 4:
        return all(count_pngs(os.path.join(LOCAL_EVAL, f"reconstructions_{c}_{run_name}")) >= expected
                   for c in CORRUPTIONS)
    if mode == 5:
        return os.path.exists(os.path.join(out_dir, f"stochastic_radii_histogram_kappa_{float(param)}.txt"))
    if mode == 6:
        return os.path.exists(os.path.join(out_dir, f"tsne_plot_kappa_{float(param)}.png"))
    if mode == 7:
        return os.path.exists(os.path.join(out_dir, f"standard_free_energy_kappa_{float(param)}.txt"))
    return False


def corruptions_done(dataset):
    expected = eval_set_size(dataset)
    return count_pngs(os.path.join(LOCAL_EVAL, "originals")) >= expected and \
        all(count_pngs(os.path.join(LOCAL_EVAL, c)) >= expected for c in CORRUPTIONS)


def metrics_done(dataset, model, dim, samples, robustness):
    """True when every trained run of this model, at this latent dim, already has its rows in the results file(s)."""
    model_dir = os.path.join(ROOT, data_name(dataset), MODEL_FOLDERS[model])
    suffix = f"_dim_{dim}_samples_{samples}"
    params = [os.path.basename(p).split("_")[2] for p in glob.glob(os.path.join(model_dir, "outputs_*"))
              if p.endswith(suffix) and os.path.exists(os.path.join(p, "epoch_log.txt"))]
    if not params:
        return False
    files = [f"robustness_results_{c}.txt" for c in CORRUPTIONS] if robustness else ["evaluation_results.txt"]
    for name in files:
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            return False
        with open(path) as f:
            done = {line.split()[0] for line in f.read().splitlines()[1:] if line.strip()}
        if not set(params) <= done:
            return False
    return True


if __name__ == "__main__":
    kind, args = sys.argv[1], sys.argv[2:]
    if kind == "step":
        dataset, model, param, dim, samples, mode = args
        ok = step_done(int(dataset), int(model), param, int(dim), int(samples), int(mode))
    elif kind == "corrupt":
        ok = corruptions_done(int(args[0]))
    elif kind == "metrics":
        ok = metrics_done(int(args[0]), int(args[1]), int(args[2]), int(args[3]), len(args) > 4)
    else:
        sys.exit(2)
    sys.exit(0 if ok else 1)
