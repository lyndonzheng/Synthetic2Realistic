import time
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()

dataset = dataloader(opt)
dataset_size = len(dataset) * opt.batchSize
print('training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps=0

for epoch in range(opt.epoch_count, opt.niter+opt.niter_decay+1):
    epoch_start_time = time.time()
    epoch_iter = 0

    # training
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters(i)

        if total_steps % opt.display_freq == 0:
            if epoch >= opt.transform_epoch:
                model.validation_target()
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save_networks('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch (epoch %d, iters %d)' % (epoch, total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    print ('End of the epoch %d / %d \t Time Take: %d sec' %
           (epoch, opt.niter + opt.niter_decay, time.time()-epoch_start_time))

    model.update_learning_rate()
