from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--dataset_mode', type=str, default='unalignwhs_test',
                            help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--name', type=str, default='ep12_new_bz16_semi_entropy_50_cps',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--results_dir', type=str, default="New_Results_45/", help='saves results here.')
        parser.add_argument('--log_dir', type=str, default='test_logs/ct2mr', help='dir to save test results')
        # parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        # parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, analyze, etc')
        parser.add_argument('--precision', type=int, default=0, help='train, val, test, analyze, etc')
        parser.add_argument('--thresholds', nargs="+", type=float, default=[0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.96,0.97,0.98], help='train, val, test, analyze, etc')
        parser.add_argument('--data_input', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', type=bool, default=True, help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=500000, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
