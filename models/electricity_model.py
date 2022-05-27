import torch
from .base_model import BaseModel
from . import networks

class ElectricityModel(BaseModel):
    def __init__(self, opt):
        """Initialize the electricity class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.loss_names = ['Sigmoid']
        self.visual_names = ['elec', 'prediction', 'flag', 'image_paths']

        self.model_names = ['G']
        self.save_name = opt.name

        # define network
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            print(self.netG)

            # define loss function
            self.criterionBCE = torch.nn.BCEWithLogitsLoss() #BCELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.netG = self.netG.to(self.device)
            self.criterionBCE = self.criterionBCE.to(self.device)
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.elec = input['elec'].to(self.device)
        self.flag = input['flag'].to(self.device)
        self.image_paths = input['path']

    def get_input(self):
        return self.elec, self.flag, self.image_paths

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.prediction = self.netG(self.elec)
    
    def backward_G(self):
        self.prediction = self.prediction.type(torch.FloatTensor)
        self.flag = self.flag.type(torch.FloatTensor)
        self.loss_Sigmoid = self.criterionBCE(self.prediction, self.flag)
        self.loss_Sigmoid.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
