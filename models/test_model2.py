from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        parser.set_defaults(no_dropout=True,norm='instance', netG='resnet_9blocks',no_lsgan=True)
        assert not is_train, 'TestModel cannot be used during training time'
        # parser.set_defaults(dataset_mode='single')
        # parser.add_argument('--model_suffix', type=str, default='_A', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        self.opt = opt
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        print(self.opt.direction)
        ## CT --> MR
        if self.opt.direction == "AtoB":
            self.visual_names = ['real_A', 'fake_B']
            self.model_names = ['G_A']  # only generator is needed.
        elif self.opt.direction == "BtoA":
            ## MR --> CT
            self.visual_names = ['real_B', 'fake_A']
            self.model_names = ['G_B']  # only generator is needed.

        if self.opt.direction == "AtoB":
            ## CT --> MR
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            print(self.netG_A)

        elif self.opt.direction == "BtoA":
            ## MR --> CT
            self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            print(self.netG_B)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        if self.opt.direction == "AtoB":
            ## CT --> MR    
            setattr(self, 'netG_A', self.netG_A)  # store netG in self.
        elif self.opt.direction == "BtoA":
            ## MR --> CT
            setattr(self, 'netG_B', self.netG_B)  # store netG in self.


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        if self.opt.direction == "AtoB":
            ## CT --> MR
            self.real_A = input['A'].to(self.device)
            self.image_paths = input['A_paths']
        elif self.opt.direction == "BtoA":
            ## MR --> CT
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass."""
        if self.opt.direction == "AtoB":
            ## CT --> MR
            self.fake_B = self.netG_A(self.real_A)  # G(A)
        elif self.opt.direction == "BtoA":
            ## MR --> CT
            self.fake_A = self.netG_B(self.real_B)  # G(A)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
