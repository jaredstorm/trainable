# Trainable: The Flexible PyTorch Training Toolbox

If you're sick of dealing with all of the boilerplate code involved in training, evaluation, visualization, and
preserving your models, then you're in luck. Trainable offers a simple, yet extensible framework to make understanding
the latest papers the *only* headache of Neural Network training.

## Installation
```bash
pip install trainable
```

## Usage
The typical workflow for trainable involves defining a callable Algorithm to describe how to train 
your network on a batch, and how you'd like to label your losses:

```python
class MSEAlgorithm(Algorithm):
    def __init__(self, eval=False, **args):
        super().__init__(eval)
        self.mse = nn.MSELoss()

    def __call__(self, model, batch, device):
        x, target = batch
        x, target = x.to(device), target.to(device)
        y = model(x)
        
        loss = self.mse(y, target)
        loss.backward()
        
        metrics = { self.key("MSE Loss"):loss.item() }
        return metrics
```

Then you simply instantiate your model, dataset, and optimizer...

```python
device = torch.device('cuda')

model = MyModel().to(device)
optim = FancyPantsOptimizer(model.parameters(), lr=1e-4)

train_data = DataLoader(SomeTrainingDataset('path/to/your/data'), batch_size=32)
test_data = DataLoader(SomeTestingDataset('path/to/your/data'), batch_size=32)
```

...and let trainable take care of the rest!
```python
trainer = Trainer(
  visualizer=MyVisualizer(),  # Typically Plotter() or Saver()
  train_alg=MyFancyAlgorithm(),
  test_alg=MyFancyAlgorithm(eval=True)
  display_freq=1,
  visualize_freq=10,
  validate_freq=10,
  autosave_freq=10,
  device=device
)

save_path = "desired/save/path/for/your/session.sesh"
trainer.start_session(model, optim, path)

trainer.name_session('Name')

trainer.describe_session("""
A beautiful detailed description of what the heck 
you were trying to accomplish with this training.
""")

metrics = trainer.train(train_data, test_data, epochs=200)
```

Plotting your data is simple as well:
```python
import matplotlib.pyplot as plt

for key in metrics:
    plt.plot(metrics[key])
    plt.show()
```

## Tunable Options
The Trainer interface gives you a nice handful of options to configure your training experience.
They include:
* **Display Frequency:** How often (in batches) information such as your training loss is updated in your progress bar.
* **Visualization Frequency:** How often (in batches) the training produces a visualization of your model's outputs. 
* **Validation Frequency:** How often (in epochs) the trainer performs validation with your test data.
* **Autosave Frequency:** How often your session is saved out to disk. 
* **Device:** On which hardware your training should occur.

## Customization
Do you want a little more granularity in how you visualize your data? Or perhaps
running an epoch with your model is a little more involved than just training
on each batch of data? Wondering why the heck pytorch doesn't have a built-in dataset for unsupervised images?
Maybe your training algorithm involves VGG? Got you covered. Check out the source for the various submodules:
* [trainable.visualize](https://github.com/hiltonjp/trainable/blob/master/trainable/visualize.py) -- for customizing visualization.
* [trainable.epoch](https://github.com/hiltonjp/trainable/blob/master/trainable/epoch.py) -- for customizing epochs.
* [trainable.data](https://github.com/hiltonjp/trainable/tree/master/trainable/data) -- for common datasets and transforms
    not found in pytorch's modules.
* [trainable.features](https://github.com/hiltonjp/trainable/tree/master/trainable/features) -- for working with intermediate
    activations and features, such as with VGG-based losses.

## Contributing
Find any other headaches in neural net training that you think you can simplify with Trainable? Feel free to make a
pull request from my [github repo](https://github.com/hiltonjp/trainable). 

## Contact
Email me anytime at [jeffhilton.code@gmail.com.]()
