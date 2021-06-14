import argparse
import pathlib

class Args(argparse.ArgumentParser):
    def __init__(self, **overrides):

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        self.add_argument('--data-path', type=pathlib.Path, required=True,
                          help='Path to the dataset')
        
        self.add_argument('--drop-prob', type=float, default=0.3, help='Dropout probability')
        self.add_argument('--num-chans', type=int, default=16, help='Number of U-Net channels')
        self.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
        self.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
        self.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        self.add_argument('--lr-step-size', type=int, default=1,
                            help='Period of learning rate decay')
        self.add_argument('--lr-gamma', type=float, default=0.98,
                            help='Multiplicative factor of learning rate decay')
        self.add_argument('--weight-decay', type=float, default=0.,
                            help='Strength of weight decay regularization')
        self.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
        self.add_argument('--device', type=str, default='cuda',
                            help='Which device to train on. Set to "cuda" to use the GPU')
        self.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                            help='Path where model and results should be saved')
        self.add_argument('--resume', action='store_true',
                            help='If set, resume the training from a previous model checkpoint. '
                                 '"--checkpoint" should be set with this')
        self.add_argument('--checkpoint', type=str,
                            help='Path to an existing checkpoint. Used along with "--resume"')
        self.add_argument('--sumpath', type=str, default='summary',
                            help='Which folder to save the event')
        self.add_argument('--gpu', type=str, default='1', help='GPU Number')
        self.add_argument('--cv', type=str, default='1', help='Cross Validation Fold')

        # Override defaults with passed overrides
        self.set_defaults(**overrides)
