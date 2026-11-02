# PICA acoustic demo

Fit a single-mode acoustic field from six complex pressure measurements, then calibrate a prediction radius under a known distribution shift. One seeded run generates the calibration data, test fields, figures of merit and a readable report. All arrays in this repository are synthetic.

## Run

Use Python 3.9 or later with NumPy. From this directory:

```sh
python -m pip install -r requirements.txt
python run_demo.py --seed 20261031 --output reproduced
```

Open `reproduced/report.html` for the summary. The directory `synthetic_data/` contains the reference outputs from the default seed. To check numerical reproduction, compare the CSV values, NPZ arrays and numerical fields of `summary.json`. The SHA-256 inventory identifies the files from each run.

## What the example computes

- The analytical pressure is `u(x) = exp((-beta + i He) x)` on 201 normalized axial coordinates. Sensors at 0.10, 0.25, 0.40, 0.55, 0.70 and 0.85 observe the field with complex noise of RMS 0.01.
- The generator draws 800 labeled source calibration fields and 400 target test fields. The bounded Gauss–Newton fit receives just the sensor coordinates and their complex pressures. Dense target references are used for scoring after fitting.
- The calibration score is the maximum complex error across one synthetic field. Known generator densities give an oracle importance ratio of 5.25 inside the target support and zero outside. The finite-sample weighted quantile includes the test-point mass at infinity.
- One separable 2-D slice illustrates magnitude SSIM using valid, uniformly weighted 11-by-11 windows, population moments and a fixed dynamic range of 1. It is built from the first target field multiplied by a 33-point transverse sine profile. Magnitude SSIM is invariant to a global phase rotation, whereas complex pressure error retains phase information.

## Files

| Output | Contents |
| --- | --- |
| `synthetic_fields.npz` | Source and target parameters, complex sensor data, reference and fitted pressures, calibration scores, weights and target radii |
| `synthetic_source_calibration.csv` | 800 source cases and oracle weights |
| `synthetic_case_results.csv` | 400 target cases with field errors, coverage, fit iterations and observation residuals |
| `summary.json`, `report.html` | Numerical results and 14 executable checks |
| `sha256.json` | Checksums for the five generated result files |

The paper reports neural-operator and hardware experiments with pointwise calibration. The scope of this release is the analytical example with whole-field scores. The study training code, checkpoint files and hardware recordings are not distributed here.

中文：运行上述两条命令即可从随机种子重建六通道合成观测、拟合声场、计算校准半径与幅值 SSIM。结果及自检在 `reproduced/report.html`。
