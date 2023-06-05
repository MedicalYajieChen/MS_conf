from .base_options import BaseOptions


class AnalyzeOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--dataset_mode', type=str, default='unalignwhs_test', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--name', type=str, default='ep10_new_bz16_semi_entropy_45_cps',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--results_dir', type=str, default="New_Results_45/", help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='analyze', help='train, val, test, analyze, etc')
        parser.add_argument('--select_epoch', type=str, default='10',
                            help='which checkpoint to generate translated images')
        parser.add_argument('--eval', type=bool, default=True, help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=500000, help='how many test images to run')
        parser.add_argument('--data_input', type=str, default='test', help='train, val, test, etc')
        parser.set_defaults(model='analyze')
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
