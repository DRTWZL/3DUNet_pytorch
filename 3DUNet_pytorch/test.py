from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import logger,common
from utils.common import get_gaussian_arr,pad_nd_image,compute_steps_for_sliding_window
import torch.nn.functional as F
from dataset.dataset_lits_test import Test_Dataset
import SimpleITK as sitk
import os
from models import UNet3D
import numpy as np
from collections import OrderedDict
import pdb

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
def Saveitk(arr,itk_data,savefile):
    newdata = sitk.GetImageFromArray(arr)
    newdata.SetOrigin(itk_data.GetOrigin())
    newdata.SetSpacing(itk_data.GetSpacing())
    newdata.SetDirection(itk_data.GetDirection())
    sitk.WriteImage(newdata, savefile) 

def Itk2array(nii_path,type):
    '''
    read nii.gz and convert to numpy array
    '''
    itk_data = sitk.ReadImage(nii_path,type)
    data = sitk.GetArrayFromImage(itk_data)
    return data,itk_data

def test(model,test_loader,patch_size,filename_list, args):
    model.eval()
    pred_save = os.path.join(args.test_data_path,"predict")
    mkdir(pred_save)

    gaussian_importance_map = get_gaussian_arr(patch_size, sigma_scale=1. / 8)
    gaussian_importance_map = torch.FloatTensor(gaussian_importance_map).cuda(0, non_blocking=True)
    step_size = 0.5
    num_classes = args.n_labels

    with torch.no_grad():
        for idx, image in tqdm(enumerate(test_loader),total=len(test_loader)):
            image = image.to(device)
            print("image shape: ",image.shape)
            new_shape = patch_size

            volume_img = image[0,:,:,:,:]
            # data padding
            data, slicer = pad_nd_image(volume_img, new_shape, "constant")
            data_shape = data.shape
            print("slide size: ",data_shape)
             
            # compute silding steps in x,y,z axis
            steps = compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
            print ("steps: ",steps)
            
            # prepare gaussian map, data, prediction and others to GPU
            # make sure all gaussian_map > 0
            add_for_nb_of_preds = gaussian_importance_map
            aggregated_results = torch.zeros([num_classes] + list(data.shape[1:]), dtype=torch.half,device=0)
            aggregated_nb_of_predictions = torch.zeros([num_classes] + list(data.shape[1:]), dtype=torch.half,device=0)
            data = data.cuda(0, non_blocking=True)
            
                       
            with torch.no_grad():
                for x in steps[0]:
                    lb_x = x
                    ub_x = x + patch_size[0]
                    for y in steps[1]:
                        lb_y = y
                        ub_y = y + patch_size[1]
                        for z in steps[2]:
                    
                            lb_z = z
                            ub_z = z + patch_size[2]
                            input_data = data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z]                    
                            input_data = input_data.cuda(0, non_blocking=True)
                            pred_s = F.softmax(model(input_data),1)[0]    
              
                                    
                            predicted_patch = pred_s*gaussian_importance_map
                            predicted_patch = predicted_patch.half()
                            aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                            aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

            slicer = tuple([slice(0, aggregated_results.shape[i]) for i in range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
            # re-padding
            aggregated_results = aggregated_results[slicer]
            aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
            class_probabilities_ori = aggregated_results / aggregated_nb_of_predictions
            
            predicted_segmentation = class_probabilities_ori.argmax(0)
            # generate result
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy().astype(np.uint8)
            pdb.set_trace()
            paid = filename_list[idx].split(".nii.gz")[0]
            imgfile = os.path.join(args.test_data_path,"img",paid+"_0.nii.gz")
            ctimg,itkdata = Itk2array(imgfile, sitk.sitkInt16)            
            savefile = os.path.join(pred_save,paid+".nii.gz")
            Saveitk(predicted_segmentation,itkdata,savefile)


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./Results', args.save)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model = UNet3D(in_dim=2, out_dim=args.n_labels,num_filters = 64).to(device)
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])

    test_log = logger.Test_Logger(save_path,"test_log")
    # data info
    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    patch_size = [28, 256, 256]   #need

    filename_list = os.listdir(os.path.join(args.test_data_path, 'label'))
    test_loader = DataLoader(dataset=Test_Dataset(args,filename_list),batch_size=1,num_workers=args.n_threads, shuffle=False)
    test(model, test_loader, patch_size, filename_list,args)


