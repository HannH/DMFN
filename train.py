import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from data.data import InpaintingDataset, ToTensor
from model.net import InpaintingModel_DFBM
from options.train_options import TrainOptions
from util.utils import getLatest
from multiprocessing import freeze_support

if __name__ == '__main__':
    config = TrainOptions().parse()

    print('loading data..')
    dataset = InpaintingDataset(config.data_file,config.dataset_path , transform=transforms.Compose([
        ToTensor()
        ]))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)

    print('data loaded..')

    print('configuring model..')
    ourModel = InpaintingModel_DFBM(opt=config)
    ourModel.print_networks()
    if config.load_model_dir != '':
        print('Loading pretrained model from {}'.format(config.load_model_dir))
        ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
        print('Loading done.')
    # ourModel = torch.nn.DataParallel(ourModel).cuda()
    print('model setting up..')
    print('training initializing..')
    writer = SummaryWriter(log_dir=config.model_folder)
    cnt = 0

    for epoch in range(config.epochs):
        freeze_support()
        for i, data in enumerate(dataloader):
            gt = data['gt'].cuda()
            # normalize to values between -1 and 1
            gt = gt / 127.5 - 1

            data_in = {'gt': gt}
            ourModel.setInput(data_in)
            ourModel.optimize_parameters()
            ourModel.update_learning_rate()

            if (i + 1) % config.viz_steps == 0:
                ret_loss = ourModel.get_current_losses()
                if config.pretrain_network is False:
                    print(
                        '[%d, %5d] G_loss: %.4f (vgg: %.4f, ae: %.4f, adv: %.4f, fm_dis: %.4f, vgg_align: %.2f, '
                        'vgg_fm: %.2f, vgg_guided: %.2f ), D_loss: %.4f, LR : %f'
                        % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_vgg'], ret_loss['G_loss_ae'],
                           ret_loss['G_loss_adv'], ret_loss['G_loss_fm_dis'], ret_loss['G_loss_vgg_align'],
                           ret_loss['G_loss_vgg_fm'], ret_loss['G_loss_vgg_guided'], ret_loss['D_loss'],ourModel.get_current_learning_rate()))
                    writer.add_scalar('adv_loss', ret_loss['G_loss_adv'], cnt)
                    writer.add_scalar('D_loss', ret_loss['D_loss'], cnt)
                    writer.add_scalar('vgg_loss', ret_loss['G_loss_vgg'], cnt)
                    writer.add_scalar('vgg_align', ret_loss['G_loss_vgg_align'], cnt)
                else:
                    print('[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f)'
                          % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_rec'], ret_loss['G_loss_ae']))

                writer.add_scalar('G_loss', ret_loss['G_loss'], cnt)
                writer.add_scalar('mae_loss', ret_loss['G_loss_ae'], cnt)

                images = ourModel.get_current_visuals_tensor()
                im_completed = vutils.make_grid(images['completed'], normalize=True, scale_each=True)
                im_input = vutils.make_grid(images['input'], normalize=True, scale_each=True)
                im_gt = vutils.make_grid(images['gt'], normalize=True, scale_each=True)
                writer.add_image('gt', im_gt, cnt)
                writer.add_image('input', im_input, cnt)
                writer.add_image('completed', im_completed, cnt)
                if (i + 1) % config.train_spe == 0:
                    print('saving model ..')
                    ourModel.save_networks(epoch + 1)
            cnt += 1
        ourModel.save_networks(epoch + 1)

    writer.export_scalars_to_json(os.path.join(config.model_folder, 'GMCNN_scalars.json'))
    writer.close()
