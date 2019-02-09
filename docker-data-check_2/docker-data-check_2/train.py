import openslide
import numpy as np
import pandas as pd
import math
from skimage.filters import threshold_otsu

PATCH_SIZE=256


print('start')

def find_patches_from_slide(slide_path, truth_path, patch_size=PATCH_SIZE,filter_non_tissue=True):
    
    slide_contains_tumor = 'pos' in slide_path
    
    ############### read_region을 위한 start, level, size를 구함 #######################
    BOUNDS_OFFSET_PROPS = (openslide.PROPERTY_NAME_BOUNDS_X, openslide.PROPERTY_NAME_BOUNDS_Y)
    BOUNDS_SIZE_PROPS = (openslide.PROPERTY_NAME_BOUNDS_WIDTH, openslide.PROPERTY_NAME_BOUNDS_HEIGHT)
    
    if slide_contains_tumor:
        with openslide.open_slide(slide_path) as slide:
            print('slide path = ',slide_path)
            start = (int(slide.properties.get('openslide.bounds-x',0)),int(slide.properties.get('openslide.bounds-y',0)))
            print('slide_properites : ', start)
            level = np.log2(patch_size) 
            level = int(level)
            
            size_scale = tuple(int(slide.properties.get(prop, l0_lim)) / l0_lim
                            for prop, l0_lim in zip(BOUNDS_SIZE_PROPS,
                            slide.dimensions))
            print('size_scale = ', size_scale)
            _l_dimensions = tuple(tuple(int(math.ceil(l_lim * scale))
                            for l_lim, scale in zip(l_size, size_scale))
                            for l_size in slide.level_dimensions)
            size = _l_dimensions[level]
            print('l_dimension_size = ', size)
            slide4 = slide.read_region(start,level,size) 

        with openslide.open_slide(truth_path) as truth:
            print('truth dimensions: ',truth.dimensions)
            z_dimensions=[]
            z_size = truth.dimensions
            z_dimensions.append(z_size)
            while z_size[0] > 1 or z_size[1] > 1:
                
                z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)
                z_dimensions.append(z_size)
            print('truth_4_dimension_size:',z_dimensions[4]) # level-4
            size = z_dimensions[4]
        with openslide.open_slide(slide_path) as slide:
            slide4 = slide.read_region(start,level,size)
            print('sldie4_dimension_size:',slide4.size)

            
    else :
        with openslide.open_slide(slide_path) as slide:
            start = (0,0)
            level = np.log2(patch_size) 
            level = int(level)
            
            size_scale = (1,1)
            _l_dimensions = tuple(tuple(int(math.ceil(l_lim * scale))
                            for l_lim, scale in zip(l_size, size_scale))
                            for l_size in slide.level_dimensions)
            
            size = _l_dimensions[level]
            
            slide4 = slide.read_region(start,level,size) 
    ####################################################################################
    
    
    # is_tissue 부분 
    slide4_grey = np.array(slide4.convert('L'))
    binary = slide4_grey > 0  # black이면 0임
    
    #  흰색영역(배경이라고 여겨지는)
    slide4_not_black = slide4_grey[slide4_grey>0]
    thresh = threshold_otsu(slide4_not_black)
    
    I, J = slide4_grey.shape
    for i in range(I):
        for j in range(J):
            if slide4_grey[i,j] > thresh :
                binary[i,j] = False
    patches = pd.DataFrame(pd.DataFrame(binary).stack())
    patches['is_tissue'] = patches[0]
    patches.drop(0, axis=1,inplace =True)
    patches['slide_path'] = slide_path

    # is_tumor 부분
    if slide_contains_tumor:
        with openslide.open_slide(truth_path) as truth:
            thumbnail_truth = truth.get_thumbnail(size) 
            print('thumbnail_truth_size:',thumbnail_truth.size)
        
        patches_y = pd.DataFrame(pd.DataFrame(np.array(thumbnail_truth.convert("L"))).stack())
        patches_y['is_tumor'] = patches_y[0] > 0
        
        # mask된 영역이 애매할 수도 있으므로
        patches_y['is_all_tumor'] = patches_y[0] == 255
        patches_y.drop(0, axis=1, inplace=True)
        samples = pd.concat([patches, patches_y], axis=1) #len(samples)
    else:
        samples = patches
        samples['is_tumor'] = False
        samples['is_all_tumor'] = False
    
    if filter_non_tissue:
        samples = samples[samples.is_tissue == True] # remove patches with no tissue #samples = samples[samples.is_tissue == True]
    
    filter_only_all_tumor = True

    if filter_only_all_tumor :
        samples['tile_loc'] = list(samples.index)
        all_tissue_samples1 = samples[samples.is_tumor==False]
        all_tissue_samples1 = all_tissue_samples1.append(samples[samples.is_all_tumor==True])
        
        all_tissue_samples1.reset_index(inplace=True, drop=True)
    
    return all_tissue_samples1

def slide_data_analysis():
    # slide, truth paths read
    image_paths = []
    with open('train.txt','r') as f:
        for line in f:
            line = line.rstrip('\n')
            image_paths.append(line)
    print('image_path # : ',len(image_paths))

    tumor_mask_paths = []
    with open('train_mask.txt','r') as f:
        for line in f:
            line = line.rstrip('\n')
            tumor_mask_paths.append(line)
    print('mask_patch # : ',len(tumor_mask_paths))
    
    # slide data anlz.
    slide_id_list = []
    num_non_tumor_list = []
    num_tumor_list = []
    num_tissues_list = []

    #image_paths
    #tumor_mask_paths
    
    #image_paths = ['data/train/image/positive/Slide001.mrxs','data/train/image/negative/Slide002.mrxs']
    #tumor_mask_paths = ['data/train/mask/positive/Slide001.png','data/train/mask/negative/Slide002.png']
    for i in range(len(image_paths)):
        slide_path = image_paths[i]
        truth_path = tumor_mask_paths[i]
        print(i,'th working','\n')
        all_tissue_samples = find_patches_from_slide(slide_path,truth_path,patch_size= 256)

        # count value id, # of tumor, # of tissues
        slide_id = int(slide_path[-8:-5])
        num_tissues = len(all_tissue_samples)
        if len(all_tissue_samples.is_tumor.value_counts()) > 1 :
            num_non_tumor, num_tumor = all_tissue_samples.is_tumor.value_counts()
            num_non_tumor = int(num_non_tumor)
            num_tumor = num_tissues - num_non_tumor
        else:
            num_non_tumor = all_tissue_samples.is_tumor.value_counts() 
            num_non_tumor = int(num_non_tumor)
            num_tumor = 0
        slide_id_list.append(slide_id)
        num_tissues_list.append(num_tissues)
        num_non_tumor_list.append(num_non_tumor)
        num_tumor_list.append(num_tumor)
        
    df = pd.DataFrame({"id":slide_id_list, "tissues":num_tissues_list, "non-tumor":num_non_tumor_list,"tumor":num_tumor_list})

    return df
    

a = slide_data_analysis()
print(a)
print('\ntissues max, mean, min = ',np.max(a.tissues),np.mean(a.tissues),np.min(a.tissues))
print('\ntumor mean ',np.mean(a.tumor))
print('\nnon-tumor mean ',np.mean(a['non-tumor']))
