import utils
import torch
import functools

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sample(config):
    model = __import__(config['model'])
    G = model.Generator(**config).cuda()

    utils.count_parameters(G)
    G.load_state_dict(torch.load('./weights/G.pth'))

    G_batch_size = max(config['G_batch_size'], config['batch_size']) 
    print("---------", G_batch_size)
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'], 
                             z_var=config['z_var'])
    G.eval()
    #Sample function
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)

    for i in range(5000):
        with torch.no_grad():
            images, labels = sample()
            print(images)

if __name__ == '__main__':
    # parse command line and run    
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())

    sample(config)