import argparse
import copy
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

matplotlib.use('Agg')
import matplotlib.pyplot as plt


TEMPERATURES = (-10, 0, 10, 25)
FEATURES = ('voltage', 'current', 'temperature', 'mean_voltage', 'mean_current')
SEED = 42


def load_sequence(path):
    data = loadmat(path, variable_names=('X', 'Y'))
    features = np.asarray(data['X'].T, dtype=np.float32)
    targets = np.asarray(data['Y'].reshape(-1, 1), dtype=np.float32)
    if features.shape != (len(targets), len(FEATURES)) or len(targets) < 5:
        raise ValueError(f'Unexpected X/Y shapes in {path.name}')
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f'Non-finite data in {path.name}')
    return features, targets


def training_data(root):
    train_x, train_y, val_x, val_y = [], [], [], []
    for temperature in TEMPERATURES:
        tag = str(temperature).replace('-', 'n')
        x, y = load_sequence(root / 'Other_Files' / f'TRAIN_LGHG2@{tag}degC_Norm_5Inputs.mat')
        split = int(0.8 * len(y))
        train_x.append(x[:split])
        train_y.append(y[:split])
        val_x.append(x[split:])
        val_y.append(y[split:])
    return tuple(np.concatenate(parts) for parts in (train_x, train_y, val_x, val_y))


def make_model():
    return nn.Sequential(
        nn.Linear(len(FEATURES), 55),
        nn.Tanh(),
        nn.Linear(55, 55),
        nn.LeakyReLU(0.3),
        nn.Linear(55, 1),
        nn.Sigmoid(),
    )


def train(train_x, train_y, val_x, val_y, epochs, batch_size):
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    mse = nn.MSELoss()
    history = []
    best_loss = float('inf')
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in torch.randperm(len(train_x)).split(batch_size):
            optimizer.zero_grad()
            loss = mse(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            val_loss = mse(model(val_x), val_y).item()
        train_loss = loss_sum / len(train_x)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError('Training produced a non-finite loss')
        history.append((epoch, train_loss, val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'Epoch {epoch:>2}/{epochs}  train MSE {train_loss:.6f}  val MSE {val_loss:.6f}', flush=True)

    model.load_state_dict(best_state)
    return model, history, best_epoch


def evaluate(model, data_dir, mean, scale):
    reports = []
    curves = []
    model.eval()
    with torch.inference_mode():
        for index, temperature in enumerate(TEMPERATURES, start=1):
            tag = str(temperature).replace('-', 'n')
            path = data_dir / 'Test' / f'{index:02d}_TEST_LGHG2@{tag}degC_Norm_(05_Inputs).mat'
            x, y = load_sequence(path)
            target = y.ravel() * 100.0
            x_tensor = torch.from_numpy((x - mean) / scale)
            prediction = model(x_tensor).numpy().ravel() * 100.0
            error = prediction - target
            reports.append({
                'temperature_c': temperature,
                'samples': len(y),
                'rmse_pp': float(np.sqrt(np.mean(error ** 2))),
                'max_absolute_error_pp': float(np.max(np.abs(error))),
            })
            curves.append((temperature, target, prediction))
    return reports, curves


def save_results(output_dir, reports, curves, history, best_epoch):
    with (output_dir / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, layout='constrained')
    for axis, (temperature, target, prediction) in zip(axes.flat, curves):
        time_minutes = np.arange(len(target)) / 60.0
        axis.plot(time_minutes, target, label='Reference', color='#263448', linewidth=1.4)
        axis.plot(time_minutes, prediction, label='PyTorch estimate', color='#b95c13', linewidth=1.1)
        axis.set(title=f'{temperature} °C', xlabel='Time (min)', ylabel='Remaining charge (%)')
        axis.set_ylim(-2, 104)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(loc='best')
    figure.suptitle('Battery state of charge on four held-out test sequences')
    figure.savefig(output_dir / 'soc_predictions.png', dpi=160)
    plt.close(figure)

    epochs, train_loss, val_loss = zip(*history)
    figure, axis = plt.subplots(figsize=(8, 4.7), layout='constrained')
    axis.semilogy(epochs, train_loss, label='Training', color='#263448')
    axis.semilogy(epochs, val_loss, label='Validation', color='#b95c13')
    axis.axvline(best_epoch, linestyle='--', color='gray', label=f'Selected epoch {best_epoch}')
    axis.set(title='Training and validation loss', xlabel='Epoch', ylabel='Mean squared error')
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_dir / 'training_loss.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a PyTorch LG HG2 charge estimator.')
    parser.add_argument('--data-dir', required=True, type=Path, help='Prepared LGHG2@n10C_to_25degC folder')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive')

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    train_x, train_y, val_x, val_y = training_data(args.data_dir)
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-6)
    print(f'Training samples: {len(train_y):,}; validation samples: {len(val_y):,}', flush=True)
    model, history, best_epoch = train(
        torch.from_numpy((train_x - mean) / scale), torch.from_numpy(train_y),
        torch.from_numpy((val_x - mean) / scale), torch.from_numpy(val_y),
        args.epochs, args.batch_size,
    )
    reports, curves = evaluate(model, args.data_dir, mean, scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_results(args.output_dir, reports, curves, history, best_epoch)
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': FEATURES,
        'feature_mean': torch.from_numpy(mean),
        'feature_scale': torch.from_numpy(scale),
        'best_epoch': best_epoch,
        'seed': SEED,
    }, args.output_dir / 'model.pt')
    print(f'Selected epoch {best_epoch} by validation loss.')
    for result in reports:
        print(f"{result['temperature_c']:>3} °C  RMSE {result['rmse_pp']:.2f} pp  "
              f"max error {result['max_absolute_error_pp']:.2f} pp")
    print(f'Wrote model, figures and metrics to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
