from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--val_freq', type=int, default=3000, help='frequency of validation')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--name', type=str, default='ep12_new_bz16_semi_entropy_45_cps', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--log_dir', type=str, default='val_logs/ct2mr', help='dir to save validation results')
        parser.add_argument('--select_epoch', type=str, default='12', help='which checkpoint to generate translated images')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--epoch_subsample', type=int, default=45, help='the starting epoch for subsampling')
        parser.add_argument('--max_reduce_rate', type=float, default=0.07, help='the max reduce rate')
        parser.add_argument('--reduce_epoch', type=int, default=30, help='how much epoch to reduce the sampling rate')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, analyze, etc')
        parser.add_argument('--data_input', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default='unalignwhs', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        # training parameters
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--num_epoch', type=int, default=100, help = 'num of total epoch')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta', type=float, default=1.0, help='maximum consistency weight')
        parser.add_argument('--lr', type=float, default=0.001, help='maximum learning rate')
        parser.add_argument('--initial_lr', type=float, default=0.001, help='initial learning rate for adam on segmentater')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='regulaizer weight to decay')
        parser.add_argument('--ema_alpha', type=float, default=0.99, help='weight of the teacher model in the early stage')
        parser.add_argument('--ema_alpha_late', type=float, default=0.999, help='weight of the teacher model in the late stage')
        parser.add_argument('--ema_late_epoch', type=int, default=50,
                            help='weight of the teacher model in the late stage')
        parser.add_argument('--up_epoch', type=int, default=50, help='learning rate up stage')
        parser.add_argument('--down_epoch', type=int, default=50, help='learning rate down stage')

        self.isTrain = True
        return parser
